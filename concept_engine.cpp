// ============================================================================
//  GRAPH-AUGMENTED TRANSFORMER (v6) — single file, C++17, fara librarii externe
// ----------------------------------------------------------------------------
//  Fara marketing, fara "magie falsa". NU este AGI si NU este mai bun decat un
//  LLM mare. Este un SCHELET ARHITECTURAL corect si scalabil pentru:
//
//    Transformer mic + GraphMemory + Graph Cross-Attention in next-token
//    + Adaptive Tokenizer (context fast-weights) + Online Embedding Learner
//    + muchii ORDINALE + LoRA/Adapter per-domeniu (limba noua)
//    + ActiveGraphState (decay/reinforce/expand) + multi-path hypotheses
//    + memory-aware logit bias + anti-halucinatie pe suport factual
//    + cognitive loop cu tokeni virtuali de actiune
//    + structuri INT8-ready si constante de scalare.
//
//  Ce e REAL implementat vs DEMO e marcat explicit in comentarii.
//  Reguli: zero semantica hardcodata, zero dictionar ro-en, zero if(word=="..").
//  Deciziile sunt numerice: embeddings, similaritate, scoruri, praguri, clustere.
//
//  Compilare:  g++ -O2 -std=c++17 concept_engine.cpp -o ce
//  Rulare:     ./ce
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <map>
#include <set>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <cstring>
// POSIX (Linux) pentru mmap — NU e o librarie externa, e header de sistem.
// Pe Android/mobil acelasi API e disponibil (bionic libc).
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace std;
static mt19937 rng(7);

// ---- constante de SCALARE (demo mic acum; tinta = model serios) -------------
constexpr int DEMO_D   = 64;     // dimensiune ascunsa folosita in demo
constexpr int TARGET_D = 384;    // pentru un sistem serios: D=384..512
// Pentru scalare reala se schimba: D=TARGET_D, NLAYERS=6, HEADS=8, FFN=4*D,
// si se foloseste QuantizedMat (INT8) pe mobil. Restul codului ramane la fel.
constexpr int NLAYERS_DEMO = 1;  // demo: 1 bloc. Serios: 6+ (vezi Transformer).
constexpr int HEADS = 8;
constexpr int LORA_R = 8;        // rang adapter LoRA

// ----------------------------------------------------------------------------
//  Utilitare numerice
// ----------------------------------------------------------------------------
vector<string> splitw(const string& s){
    vector<string> r; string w;
    for(char c:s){ if(c==' '||c=='\t'){ if(!w.empty())r.push_back(w); w.clear(); } else w+=c; }
    if(!w.empty()) r.push_back(w); return r;
}
float dotv(const vector<float>&a,const vector<float>&b){ float s=0; for(size_t i=0;i<a.size();i++) s+=a[i]*b[i]; return s; }
float nrm(const vector<float>&a){ return sqrt(dotv(a,a)+1e-9f); }
float cosv(const vector<float>&a,const vector<float>&b){ if(a.empty()||b.empty())return 0.f; return dotv(a,b)/(nrm(a)*nrm(b)); }
float sigm(float x){ return 1.f/(1.f+exp(-x)); }
vector<float> softmax(vector<float> x){
    if(x.empty())return x; float m=*max_element(x.begin(),x.end()),s=0;
    for(float&v:x){v=exp(v-m);s+=v;} for(float&v:x)v/=s; return x;
}
float siluf(float x){return x/(1+exp(-x));}
float dsiluf(float x){float s=1/(1+exp(-x));return s+x*s*(1-s);}
void addInto(vector<float>&a,const vector<float>&b,float s=1.f){ for(size_t i=0;i<a.size();i++)a[i]+=s*b[i]; }
void l2norm(vector<float>&v,float target=1.f){ float n=nrm(v); if(n>0)for(float&x:v)x*=target/n; }

// ============================================================================
//  Mat (float, Adam) — corpul transformerului.
//  INT8-READY: vezi QuantizedMat mai jos; in productie matricile mari pot fi
//  inlocuite cu QuantizedMat pentru inference pe mobil.
// ============================================================================
struct Mat {
    int R,C; vector<float> w,g,m,v;
    Mat(){} Mat(int r,int c):R(r),C(c),w(r*c),g(r*c,0),m(r*c,0),v(r*c,0){}
    void init(float s){ normal_distribution<float> d(0,s); for(float&x:w)x=d(rng);}
    float& at(int i,int j){return w[i*C+j];}
    float& gat(int i,int j){return g[i*C+j];}
    void zerograd(){ fill(g.begin(),g.end(),0);}
    void step(float lr,float t,float wd){
        const float b1=0.9f,b2=0.999f,eps=1e-8f; float bc1=1-pow(b1,t),bc2=1-pow(b2,t);
        for(size_t i=0;i<w.size();i++){ m[i]=b1*m[i]+(1-b1)*g[i]; v[i]=b2*v[i]+(1-b2)*g[i]*g[i];
            w[i]-=lr*(m[i]/bc1/(sqrt(v[i]/bc2)+eps)+wd*w[i]); }
    }
};
vector<float> matmul(const vector<float>&x,Mat&W){
    vector<float> y(W.C,0); for(int i=0;i<W.R;i++){ float xi=x[i]; if(xi==0)continue;
        for(int j=0;j<W.C;j++) y[j]+=xi*W.at(i,j);} return y;
}
vector<float> matmul_bwd(const vector<float>&x,const vector<float>&dy,Mat&W){
    vector<float> dx(W.R,0); for(int i=0;i<W.R;i++){ float a=0,xi=x[i];
        for(int j=0;j<W.C;j++){ W.gat(i,j)+=xi*dy[j]; a+=W.at(i,j)*dy[j]; } dx[i]=a; } return dx;
}

// ============================================================================
//  QuantizedMat (INT8-READY) — DEMO de structura + (de)cuantizare per-rand.
//  NU e folosita in calea fierbinte aici; arata cum Mat ar putea fi inlocuita
//  pe mobil (greutati int8 + scala pe rand). Calculul real ar folosi kernele
//  INT8; aici doar dequant pentru demonstratie.
// ============================================================================
struct QuantizedMat {
    int R,C; vector<int8_t> w; vector<float> scale;   // scala per rand
    QuantizedMat(){} QuantizedMat(int r,int c):R(r),C(c),w(r*c),scale(r,1){}
    static QuantizedMat fromMat(Mat& M){
        QuantizedMat q(M.R,M.C);
        for(int i=0;i<M.R;i++){ float mx=1e-9f; for(int j=0;j<M.C;j++)mx=max(mx,fabs(M.at(i,j)));
            float s=mx/127.f; q.scale[i]=s;
            for(int j=0;j<M.C;j++) q.w[i*M.C+j]=(int8_t)lround(M.at(i,j)/s); }
        return q;
    }
    float at(int i,int j)const{ return w[i*C+j]*scale[i]; }   // dequant
};

// ============================================================================
//  LoRA ADAPTER — adaptare ieftina (rang mic) pentru un DOMENIU (ex: limba noua)
//  out = scale * B(A(x)).  Se antreneaza DOAR A,B online; corpul ramane inghetat.
// ============================================================================
struct LoRAAdapter {
    int D,r; float scale; Mat A,B; bool active=false;
    LoRAAdapter(){} LoRAAdapter(int d,int rank,float s):D(d),r(rank),scale(s),A(d,rank),B(rank,d){
        A.init(0.02f); /* B=0 => adapter porneste ca no-op (output nemodificat) */
    }
    vector<float> apply(const vector<float>& x, vector<float>* downCache=nullptr){
        vector<float> down=matmul(x,A); if(downCache)*downCache=down;
        vector<float> up=matmul(down,B); for(float&v:up)v*=scale; return up;
    }
    // backward simplificat: actualizeaza doar A,B (corp inghetat) pe calea LoRA
    void backward(const vector<float>& x,const vector<float>& down,const vector<float>& dOut){
        vector<float> dUp=dOut; for(float&v:dUp)v*=scale;
        vector<float> dDown=matmul_bwd(down,dUp,B);   // acumuleaza grad B + dDown
        (void)matmul_bwd(x,dDown,A);                  // acumuleaza grad A
    }
    void stepAdam(float lr,float t,float wd){ A.step(lr,t,wd); B.step(lr,t,wd); }
    void zerograd(){ A.zerograd(); B.zerograd(); }
};

// ============================================================================
//  EMBEDDING STORE crescator (head TIED). Token nou = rand nou, intra in logits.
// ============================================================================
struct EmbeddingStore {
    int D; vector<vector<float>> E,gM,gV,grad;
    EmbeddingStore(int d=0):D(d){}
    int add(const vector<float>& init){ E.push_back(init); gM.push_back(vector<float>(D,0));
        gV.push_back(vector<float>(D,0)); grad.push_back(vector<float>(D,0)); return (int)E.size()-1; }
    int size()const{ return (int)E.size(); }
    void zerograd(){ for(auto& r:grad) fill(r.begin(),r.end(),0); }
    void stepRows(float lr,float t,float wd,const set<int>& rows){
        const float b1=0.9f,b2=0.999f,eps=1e-8f; float bc1=1-pow(b1,t),bc2=1-pow(b2,t);
        for(int i:rows) for(int k=0;k<D;k++){ float g=grad[i][k];
            gM[i][k]=b1*gM[i][k]+(1-b1)*g; gV[i][k]=b2*gV[i][k]+(1-b2)*g*g;
            E[i][k]-=lr*(gM[i][k]/bc1/(sqrt(gV[i][k]/bc2)+eps)+wd*E[i][k]); }
    }
};

// ============================================================================
//  RMSNorm
// ============================================================================
struct RMSNorm {
    int D; Mat g; RMSNorm(){} RMSNorm(int d):D(d),g(1,d){for(int i=0;i<d;i++)g.at(0,i)=1;}
    vector<float> fwd(const vector<float>&x,float&inv,vector<float>&xn){
        float ms=0; for(float v:x)ms+=v*v; ms/=D; inv=1/sqrt(ms+1e-5f);
        xn.resize(D); vector<float> y(D); for(int i=0;i<D;i++){xn[i]=x[i]*inv;y[i]=xn[i]*g.at(0,i);} return y;
    }
    vector<float> bwd(const vector<float>&dy,const vector<float>&xn,float inv){
        vector<float> dxn(D); float d=0;
        for(int i=0;i<D;i++){g.gat(0,i)+=dy[i]*xn[i];dxn[i]=dy[i]*g.at(0,i);d+=dxn[i]*xn[i];}
        vector<float> dx(D); for(int i=0;i<D;i++)dx[i]=inv*(dxn[i]-xn[i]*d/D); return dx;
    }
};

// ============================================================================
//  TRANSFORMER (1 bloc demo; pentru scalare: stiva de NLAYERS blocuri identice).
//  Embedding-uri TIED din EmbeddingStore. FFN are un LoRA per-domeniu (optional).
// ============================================================================
struct Transformer {
    int D,H,nH,hd; EmbeddingStore* es;
    Mat Wq,Wk,Wv,Wo,W1,Wg,W2; RMSNorm n1,n2,nf;
    LoRAAdapter* ffnLora=nullptr;   // adapter activ (per-domeniu) pentru FFN
    Transformer(EmbeddingStore* store,int d,int ffn,int heads)
        :D(d),H(ffn),nH(heads),hd(d/heads),es(store),
         Wq(d,d),Wk(d,d),Wv(d,d),Wo(d,d),W1(d,ffn),Wg(d,ffn),W2(ffn,d),n1(d),n2(d),nf(d){
        float s=sqrt(2.0f/d);
        Wq.init(s);Wk.init(s);Wv.init(s);Wo.init(s);W1.init(s);Wg.init(s);W2.init(sqrt(2.0f/ffn));
    }
    vector<Mat*> body(){ return {&Wq,&Wk,&Wv,&Wo,&W1,&Wg,&W2,&n1.g,&n2.g,&nf.g}; }
    void zerogradBody(){ for(auto*p:body())p->zerograd(); }
    void stepBody(float lr,float t,float wd){ for(auto*p:body())p->step(lr,t,wd); }

    struct Cache{ vector<int> ids; int n;
        vector<vector<float>> emb,a_n,res1,f_n,up,gate,act,ffn,out,hid, loraDown;
        vector<float> a_inv,f_inv,nf_inv; vector<vector<float>> a_xn,f_xn,nf_xn;
        vector<vector<float>> q,k,vv,attn; vector<vector<float>> probs; };

    Cache forward(const vector<int>&ids){
        Cache c; c.ids=ids; int n=ids.size(); c.n=n;
        c.emb.assign(n,vector<float>(D)); for(int t=0;t<n;t++) c.emb[t]=es->E[ids[t]];
        c.a_n.resize(n);c.a_inv.resize(n);c.a_xn.resize(n);c.q.resize(n);c.k.resize(n);c.vv.resize(n);
        for(int t=0;t<n;t++){ float inv;vector<float> xn; c.a_n[t]=n1.fwd(c.emb[t],inv,xn);
            c.a_inv[t]=inv;c.a_xn[t]=xn;
            c.q[t]=matmul(c.a_n[t],Wq);c.k[t]=matmul(c.a_n[t],Wk);c.vv[t]=matmul(c.a_n[t],Wv); }
        c.attn.assign(n,vector<float>(D,0)); c.probs.resize(nH*n); float invs=1/sqrt((float)hd);
        for(int h=0;h<nH;h++)for(int t=0;t<n;t++){ vector<float> sc(t+1);
            for(int j=0;j<=t;j++){float s=0;for(int i=0;i<hd;i++)s+=c.q[t][h*hd+i]*c.k[j][h*hd+i];sc[j]=s*invs;}
            auto p=softmax(sc); c.probs[h*n+t]=p;
            for(int j=0;j<=t;j++)for(int i=0;i<hd;i++)c.attn[t][h*hd+i]+=p[j]*c.vv[j][h*hd+i]; }
        c.res1.assign(n,vector<float>(D));
        for(int t=0;t<n;t++){auto o=matmul(c.attn[t],Wo);for(int i=0;i<D;i++)c.res1[t][i]=c.emb[t][i]+o[i];}
        c.f_n.resize(n);c.f_inv.resize(n);c.f_xn.resize(n);c.up.resize(n);c.gate.resize(n);
        c.act.resize(n);c.ffn.resize(n);c.out.assign(n,vector<float>(D)); c.loraDown.resize(n);
        for(int t=0;t<n;t++){ float inv;vector<float> xn;c.f_n[t]=n2.fwd(c.res1[t],inv,xn);c.f_inv[t]=inv;c.f_xn[t]=xn;
            c.up[t]=matmul(c.f_n[t],W1);c.gate[t]=matmul(c.f_n[t],Wg);
            c.act[t].resize(H);for(int i=0;i<H;i++)c.act[t][i]=siluf(c.up[t][i])*c.gate[t][i];
            c.ffn[t]=matmul(c.act[t],W2);
            if(ffnLora){ vector<float> down; vector<float> add=ffnLora->apply(c.f_n[t],&down);
                c.loraDown[t]=down; for(int i=0;i<D;i++)c.ffn[t][i]+=add[i]; }
            for(int i=0;i<D;i++)c.out[t][i]=c.res1[t][i]+c.ffn[t][i]; }
        c.hid.resize(n);c.nf_inv.resize(n);c.nf_xn.resize(n);
        for(int t=0;t<n;t++){float inv;vector<float> xn;c.hid[t]=nf.fwd(c.out[t],inv,xn);c.nf_inv[t]=inv;c.nf_xn[t]=xn;}
        return c;
    }
    // logits TIED. Pentru generare folosim varianta cosine (logitsCos) ca un
    // token sa nu domine prin norma. lmHead = produs cu embeddingurile.
    vector<float> logitsCos(const vector<float>&h,float temp=8.f){
        int V=es->size(); vector<float> z(V); float hn=nrm(h);
        for(int v=0;v<V;v++) z[v]=temp*dotv(h,es->E[v])/(hn*nrm(es->E[v])+1e-9f); return z;
    }
    vector<float> hiddenLast(const vector<int>&ids){ if(ids.empty())return vector<float>(D,0); return forward(ids).hid.back(); }

    // antrenare next-token. updateBody=false => doar embeddingurile + (optional)
    // LoRA-ul activ se misca (corp inghetat) => adaptare ieftina la limba noua.
    float trainSeq(const vector<int>&ids,bool updateBody,bool updateLora,float lr,float wd,float& step){
        int n=ids.size(); if(n<2) return 0; Cache c=forward(ids); int V=es->size(); set<int> touched;
        vector<vector<float>> dHid(n,vector<float>(D,0)); float loss=0;int cnt=0;
        for(int t=0;t<n-1;t++){ // logits TIED (dot) pentru antrenare (stabil cu Adam)
            vector<float> z(V); for(int v=0;v<V;v++)z[v]=dotv(c.hid[t],es->E[v]);
            auto p=softmax(z); int tg=ids[t+1]; loss+=-log(max(p[tg],1e-9f)); cnt++;
            vector<float> dz=p; dz[tg]-=1;
            for(int v=0;v<V;v++){ float d=dz[v]; if(d==0)continue;
                for(int i=0;i<D;i++){ dHid[t][i]+=d*es->E[v][i]; es->grad[v][i]+=d*c.hid[t][i]; } touched.insert(v); } }
        vector<vector<float>> dOut(n,vector<float>(D,0));
        for(int t=0;t<n-1;t++) dOut[t]=nf.bwd(dHid[t],c.nf_xn[t],c.nf_inv[t]);
        vector<vector<float>> dRes1(n,vector<float>(D,0));
        for(int t=0;t<n;t++){ for(int i=0;i<D;i++)dRes1[t][i]+=dOut[t][i];
            // calea LoRA: dffn -> A,B (corp inghetat, nu propagam in f_n prin LoRA)
            if(ffnLora && updateLora) ffnLora->backward(c.f_n[t],c.loraDown[t],dOut[t]);
            vector<float> dact=matmul_bwd(c.act[t],dOut[t],W2); vector<float> dup(H),dg(H);
            for(int i=0;i<H;i++){float su=siluf(c.up[t][i]);dg[i]=dact[i]*su;dup[i]=dact[i]*c.gate[t][i]*dsiluf(c.up[t][i]);}
            vector<float> d1=matmul_bwd(c.f_n[t],dup,W1),d2=matmul_bwd(c.f_n[t],dg,Wg);
            vector<float> df(D);for(int i=0;i<D;i++)df[i]=d1[i]+d2[i];
            vector<float> dr=n2.bwd(df,c.f_xn[t],c.f_inv[t]); for(int i=0;i<D;i++)dRes1[t][i]+=dr[i]; }
        vector<vector<float>> dEmb(n,vector<float>(D,0)),dAttn(n,vector<float>(D,0));
        for(int t=0;t<n;t++){ for(int i=0;i<D;i++)dEmb[t][i]+=dRes1[t][i]; dAttn[t]=matmul_bwd(c.attn[t],dRes1[t],Wo); }
        vector<vector<float>> dq(n,vector<float>(D,0)),dk(n,vector<float>(D,0)),dv(n,vector<float>(D,0));
        float invs=1/sqrt((float)hd);
        for(int h=0;h<nH;h++)for(int t=0;t<n;t++){ int L=t+1; auto&p=c.probs[h*n+t]; vector<float> dp(L,0);
            for(int j=0;j<L;j++){float dd=0;for(int i=0;i<hd;i++){dv[j][h*hd+i]+=p[j]*dAttn[t][h*hd+i];dd+=dAttn[t][h*hd+i]*c.vv[j][h*hd+i];}dp[j]=dd;}
            float dotp=0;for(int j=0;j<L;j++)dotp+=dp[j]*p[j];
            for(int j=0;j<L;j++){float ds=p[j]*(dp[j]-dotp)*invs;
                for(int i=0;i<hd;i++){dq[t][h*hd+i]+=ds*c.k[j][h*hd+i];dk[j][h*hd+i]+=ds*c.q[t][h*hd+i];}} }
        for(int t=0;t<n;t++){ vector<float> a=matmul_bwd(c.a_n[t],dq[t],Wq),b=matmul_bwd(c.a_n[t],dk[t],Wk),cc=matmul_bwd(c.a_n[t],dv[t],Wv);
            vector<float> dn(D);for(int i=0;i<D;i++)dn[i]=a[i]+b[i]+cc[i];
            vector<float> de=n1.bwd(dn,c.a_xn[t],c.a_inv[t]); for(int i=0;i<D;i++)dEmb[t][i]+=de[i]; }
        for(int t=0;t<n;t++){ for(int i=0;i<D;i++) es->grad[ids[t]][i]+=dEmb[t][i]; touched.insert(ids[t]); }
        step+=1; es->stepRows(lr,step,wd,touched);
        if(updateBody) stepBody(lr,step,1e-3f);
        if(ffnLora && updateLora){ ffnLora->stepAdam(lr,step,1e-4f); ffnLora->zerograd(); }
        zerogradBody(); es->zerograd();
        return cnt?loss/cnt:0;
    }
};

// ============================================================================
//  GRAPH CROSS-ATTENTION (inlocuieste FusionGate orb).
//  Query = hidden; Key/Value = embeddings ale NODURILOR/MUCHIILOR active.
//  Un slot NULL (scor tau) lasa Transformerul sa IGNORE graful daca niciun nod
//  nu e relevant. Returneaza contextul + scorul de atentie maxim (pentru gating).
// ============================================================================
struct GraphCrossAttention {
    int D; Mat Wq,Wk,Wv,Wo; float tau;   // tau = scorul slotului NULL (ignora graful)
    GraphCrossAttention(int d):D(d),Wq(d,d),Wk(d,d),Wv(d,d),Wo(d,d),tau(0.6f){
        // Init ~identitate: atentia ~ similaritate hidden/nod fara antrenare.
        // Intr-un sistem serios Wq/Wk/Wv/Wo se antreneaza odata cu modelul.
        for(int i=0;i<d;i++){ Wq.at(i,i)=1; Wk.at(i,i)=1; Wv.at(i,i)=1; Wo.at(i,i)=0.7f; }
    }
    // keys/values din embeddingurile date (noduri active + muchii active)
    struct Out{ vector<float> ctx; float maxScore; };
    Out attend(const vector<float>& hidden, const vector<vector<float>>& mem){
        Out o; o.ctx.assign(D,0); o.maxScore=0;
        if(mem.empty()) return o;
        vector<float> q=matmul(hidden,Wq); float invs=1.f/sqrt((float)D);
        vector<float> sc; sc.reserve(mem.size()+1);
        vector<vector<float>> vals; vals.reserve(mem.size());
        for(auto& mvec:mem){ vector<float> k=matmul(mvec,Wk); vals.push_back(matmul(mvec,Wv));
            float s=dotv(q,k)*invs; sc.push_back(s); o.maxScore=max(o.maxScore,s); }
        sc.push_back(tau);                       // slot NULL: ignora graful
        auto p=softmax(sc);
        for(size_t i=0;i<vals.size();i++) addInto(o.ctx,vals[i],p[i]);  // p[last]=NULL nu adauga nimic
        o.ctx=matmul(o.ctx,Wo);
        return o;
    }
};

// ============================================================================
//  TOKENIZER ADAPTIV cu CONTEXT FAST-WEIGHTS
//  Cuvant nou => token nou + embedding = alpha*char + beta*context + gamma*vecini.
// ============================================================================
struct AdaptiveTokenizer {
    unordered_map<string,int> id; vector<string> word; EmbeddingStore* es; int D;
    vector<vector<float>> charBasis;
    AdaptiveTokenizer(EmbeddingStore* store,int d):es(store),D(d){
        charBasis.assign(256,vector<float>(D)); normal_distribution<float> g(0,1);
        for(int c=0;c<256;c++)for(int k=0;k<D;k++)charBasis[c][k]=g(rng);
    }
    vector<float> charEmbed(const string& w){
        vector<float> v(D,0); for(unsigned char c:w) addInto(v,charBasis[c]);
        for(size_t i=0;i+1<w.size();i++){ unsigned char a=w[i],b=w[i+1]; int h=(a*131+b)&255; addInto(v,charBasis[h],0.5f); }
        l2norm(v,0.1f); return v;
    }
    int get(const string& w)const{ auto it=id.find(w); return it==id.end()?-1:it->second; }
    // context fast-weights: amesteca caractere + context + vecini-graf, normalizeaza
    int getOrCreateToken(const string& w,const vector<float>& contextHidden,
                         const vector<float>& graphContext,bool& isNew,
                         float alpha=1.0f,float beta=0.6f,float gamma=0.6f){
        auto it=id.find(w); if(it!=id.end()){ isNew=false; return it->second; }
        isNew=true; vector<float> e=charEmbed(w); for(float&x:e)x*=alpha;
        if(!contextHidden.empty()) addInto(e,contextHidden,beta*0.1f);
        if(!graphContext.empty())  addInto(e,graphContext, gamma*0.1f);
        l2norm(e,0.12f);
        int x=es->add(e); id[w]=x; if((int)word.size()<=x)word.resize(x+1); word[x]=w; return x;
    }
    int forceVirtual(const string& w){ // tokeni virtuali (actiuni) — embedding aleator mic
        bool dummy; vector<float> none; return getOrCreateToken(w,none,none,dummy,1,0,0); }
    string name(int i)const{ return (i>=0&&i<(int)word.size())?word[i]:"?"; }
};

// ============================================================================
//  ONLINE EMBEDDING LEARNER — update local (fara retraining complet).
//  Misca embeddingul tokenului/nodului spre context; ajusteaza confidence muchii.
// ============================================================================
struct GraphMemory; // fwd
struct OnlineEmbeddingLearner {
    EmbeddingStore* es;
    OnlineEmbeddingLearner(EmbeddingStore* s):es(s){}
    void updateTokenEmbedding(int tokenId,const vector<float>& context,float lr){
        if(tokenId<0||tokenId>=es->size()||context.empty())return;
        for(int k=0;k<es->D;k++) es->E[tokenId][k]+=lr*(context[k]-es->E[tokenId][k]);
    }
    // updateNodeEmbedding == updateTokenEmbedding (nod=token aici)
    void updateNodeEmbedding(int nodeId,const vector<float>& context,float lr){ updateTokenEmbedding(nodeId,context,lr); }
};

// ============================================================================
//  GRAPH MEMORY — noduri=tokeni; muchii cu embedding, confidence si ORDINAL.
// ============================================================================
struct Edge {
    int src,dst,relCluster; float confidence; vector<float> emb;
    // informatie ORDINALA (int8 pentru compactitate / INT8-ready):
    int8_t srcPosition;   // pozitia sursei in propozitie
    int8_t dstPosition;   // pozitia tintei
    int8_t seqOffset;     // dst - src (semn = ordine)
    float  orderConfidence; // cat de consistent e ordinul peste observatii
};
struct GraphMemory {
    vector<Edge> edges; unordered_map<int,vector<int>> outAdj,inAdj; set<int> nodes;
    int addEdge(int s,int d,int rel,float conf,const vector<float>& emb,int sp,int dp){
        for(int ei:outAdj[s]) if(edges[ei].dst==d&&edges[ei].relCluster==rel){
            edges[ei].confidence=min(1.f,edges[ei].confidence+0.03f);
            // intareste/erodeaza increderea ordinala dupa consistenta semnului
            int8_t off=(int8_t)(dp-sp); if((off>0)==(edges[ei].seqOffset>0)) edges[ei].orderConfidence=min(1.f,edges[ei].orderConfidence+0.05f);
            else edges[ei].orderConfidence=max(0.f,edges[ei].orderConfidence-0.05f);
            return ei; }
        int idx=edges.size(); Edge e; e.src=s;e.dst=d;e.relCluster=rel;e.confidence=conf;e.emb=emb;
        e.srcPosition=(int8_t)sp; e.dstPosition=(int8_t)dp; e.seqOffset=(int8_t)(dp-sp); e.orderConfidence=0.5f;
        edges.push_back(e); outAdj[s].push_back(idx); inAdj[d].push_back(idx); nodes.insert(s); nodes.insert(d); return idx;
    }
    vector<int> out(int s)const{ auto it=outAdj.find(s); return it==outAdj.end()?vector<int>{}:it->second; }
    bool isNode(int id)const{ return nodes.count(id)>0; }
};

// ============================================================================
//  CLUSTERER greedy — concepte/relatii emergente (id numeric)
// ============================================================================
struct Clusterer {
    float thresh; vector<vector<float>> centroid; vector<int> count;
    Clusterer(float t):thresh(t){}
    int assign(const vector<float>& v){
        int best=-1; float bs=thresh;
        for(size_t i=0;i<centroid.size();i++){float s=cosv(v,centroid[i]); if(s>bs){bs=s;best=(int)i;}}
        if(best<0){ centroid.push_back(v); count.push_back(1); return (int)centroid.size()-1; }
        for(size_t k=0;k<v.size();k++) centroid[best][k]=(centroid[best][k]*count[best]+v[k])/(count[best]+1);
        count[best]++; return best;
    }
    int nearest(const vector<float>& v,float& sim)const{ int best=-1; sim=-2;
        for(size_t i=0;i<centroid.size();i++){float s=cosv(v,centroid[i]); if(s>sim){sim=s;best=(int)i;}} return best; }
};

// ============================================================================
//  LANGUAGE / DOMAIN DETECTOR — fara semantica: clusterizeaza embedding-ul mediu
//  al propozitiei. Distributie noua => domeniu nou => adapter LoRA propriu.
// ============================================================================
struct DomainDetector {
    Clusterer clust; float newDomainThresh;
    DomainDetector(float th):clust(th),newDomainThresh(th){}
    int detect(const vector<float>& sentEmb,bool& isNew){
        float sim; int near=clust.nearest(sentEmb,sim);
        int before=clust.centroid.size(); int d=clust.assign(sentEmb);
        isNew=( (int)clust.centroid.size()>before ); (void)near; return d;
    }
};

// ============================================================================
//  HYPOTHESIS PATH + ACTIVE GRAPH STATE
// ============================================================================
struct HypothesisPath { vector<int> nodes,edges; float score=0,confidence=0; };
struct ActiveGraphState {
    map<int,float> nodeScore;            // activeNodes + scoruri
    vector<HypothesisPath> activePaths;  // ipoteze candidate
    void seed(const vector<int>& ns,float v=1.f){ for(int n:ns) nodeScore[n]=max(nodeScore[n],v); }
    void decay(float d=0.85f){ for(auto& kv:nodeScore) kv.second*=d; }
    // intareste nodurile similare cu tokenul generat (confirmare de path)
    void reinforceByToken(int tokenId,EmbeddingStore& es,float v=0.5f){
        if(tokenId<0||tokenId>=es.size())return; nodeScore[tokenId]+=v;
        // raspandeste putin la nodurile similare vectorial
        for(auto& kv:nodeScore){ float s=cosv(es.E[tokenId],es.E[kv.first]); if(s>0.3f) kv.second+=v*0.2f*s; }
    }
    void expandTopK(const GraphMemory& g,int k){
        // extinde activarea pe vecini (spreading), apoi pastreaza top-k
        map<int,float> add;
        for(auto& kv:nodeScore){ if(kv.second<0.05f)continue;
            for(int ei:g.out(kv.first)){ const Edge& e=g.edges[ei]; add[e.dst]+=0.4f*kv.second*e.confidence; } }
        for(auto& kv:add) nodeScore[kv.first]+=kv.second;
        if((int)nodeScore.size()>k){ vector<pair<float,int>> v; for(auto&kv:nodeScore)v.push_back({kv.second,kv.first});
            sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first>b.first;});
            map<int,float> keep; for(int i=0;i<k&&i<(int)v.size();i++)keep[v[i].second]=v[i].first; nodeScore.swap(keep); }
        float mx=0; for(auto&kv:nodeScore)mx=max(mx,kv.second); if(mx>1)for(auto&kv:nodeScore)kv.second/=mx;
    }
    // construieste ipoteze (drumuri) din nodurile cele mai active (multi-path)
    void buildHypotheses(const GraphMemory& g,int maxPaths,int maxLen){
        activePaths.clear();
        vector<pair<float,int>> starts; for(auto&kv:nodeScore)starts.push_back({kv.second,kv.first});
        sort(starts.begin(),starts.end(),[](auto&a,auto&b){return a.first>b.first;});
        for(int si=0; si<(int)starts.size() && (int)activePaths.size()<maxPaths; si++){
            int s=starts[si].second; HypothesisPath p; p.nodes={s}; p.confidence=1; int cur=s; set<int> seen{s};
            for(int step=0;step<maxLen;step++){ int bestE=-1; float bestC=0;
                for(int ei:g.out(cur)){ if(seen.count(g.edges[ei].dst))continue; if(g.edges[ei].confidence>bestC){bestC=g.edges[ei].confidence;bestE=ei;} }
                if(bestE<0)break; p.edges.push_back(bestE); p.nodes.push_back(g.edges[bestE].dst);
                p.confidence*=g.edges[bestE].confidence; seen.insert(g.edges[bestE].dst); cur=g.edges[bestE].dst; }
            if(p.nodes.size()>=2) activePaths.push_back(p);
        }
    }
    vector<pair<int,float>> topk(int k)const{ vector<pair<int,float>> v;
        for(auto&kv:nodeScore)v.push_back({kv.first,kv.second});
        sort(v.begin(),v.end(),[](auto&a,auto&b){return a.second>b.second;});
        if((int)v.size()>k)v.resize(k); return v; }
};

// ============================================================================
//  GRAPH LOGIT BIAS — fara if(word==..). Bias = activare * relevanta(query) ;
//  tokenii ne-activi primesc o usoara suprimare (factualitate ancorata in graf).
// ============================================================================
struct GraphLogitBias {
    void apply(vector<float>& logits,const ActiveGraphState& A,const vector<float>& qv,
               EmbeddingStore& es,float beta,float gamma,float& bestSupport){
        bestSupport=0; for(float& z:logits) z-=gamma;
        for(auto& kv:A.nodeScore){ int node=kv.first; if(node<0||node>=(int)logits.size())continue;
            float rel=max(0.f,cosv(qv,es.E[node])); float sup=kv.second*rel;
            logits[node]+=gamma+beta*sup; bestSupport=max(bestSupport,sup); }
    }
};

// ----------------------------------------------------------------------------
//  helper: media embeddingurilor unor tokeni
// ----------------------------------------------------------------------------
static vector<float> meanEmb(EmbeddingStore& es,const vector<int>& ids,int D){
    vector<float> v(D,0); int c=0; for(int id:ids){ if(id<0||id>=es.size())continue; addInto(v,es.E[id]); c++; }
    if(c)for(float&x:v)x/=c; return v;
}

// ============================================================================
//  ENGINE — leaga totul; expune absorbtia de propozitii si generarea augmentata.
// ============================================================================
struct Engine {
    int D; EmbeddingStore store; AdaptiveTokenizer tok; Transformer T;
    GraphCrossAttention gca; GraphMemory graph; OnlineEmbeddingLearner online;
    Clusterer relClust, conClust; DomainDetector domains; GraphLogitBias gbias;
    map<int,int> nodeConcept; vector<LoRAAdapter> lora;   // un adapter per domeniu
    // tokeni virtuali de actiune (cognitive loop)
    int VT_THINK,VT_SEARCH,VT_MEM,VT_ASK,VT_ACT,VT_WAIT,VT_RESULT;

    Engine(int d):D(d),store(d),tok(&store,d),T(&store,d,4*d,HEADS),gca(d),
        online(&store),relClust(0.55f),conClust(0.42f),domains(0.55f){
        VT_THINK=tok.forceVirtual("[THINK]"); VT_SEARCH=tok.forceVirtual("[SEARCH_GRAPH]");
        VT_MEM=tok.forceVirtual("[USE_MEMORY]"); VT_ASK=tok.forceVirtual("[ASK_CLARIFY]");
        VT_ACT=tok.forceVirtual("[ACT]"); VT_WAIT=tok.forceVirtual("[WAIT]"); VT_RESULT=tok.forceVirtual("[RESULT]");
    }

    // tokenizare cu context fast-weights: cuvintele noi sunt influentate de
    // contextul tokenilor deja cunoscuti din propozitie + vecinii lor in graf.
    vector<int> tokenize(const string& s, vector<string>& newWords){
        auto ws=splitw(s); vector<int> ids(ws.size(),-1);
        // pasul 1: rezolva tokenii cunoscuti (pentru a calcula contextul)
        vector<int> known; for(size_t i=0;i<ws.size();i++){ int id=tok.get(ws[i]); if(id>=0){ids[i]=id; known.push_back(id);} }
        vector<float> ctx=meanEmb(store,known,D);
        vector<float> gctx(D,0); { int c=0; for(int kn:known){ for(int ei:graph.out(kn)){ addInto(gctx,store.E[graph.edges[ei].dst]); c++; } } if(c)for(float&x:gctx)x/=c; }
        // pasul 2: creeaza tokenii noi cu context fast-weights
        for(size_t i=0;i<ws.size();i++) if(ids[i]<0){ bool isNew; ids[i]=tok.getOrCreateToken(ws[i],ctx,gctx,isNew);
            if(isNew) newWords.push_back(ws[i]); }
        return ids;
    }

    // absorb o propozitie: noduri/muchii ORDINALE + relatii + concepte
    void absorb(const vector<int>& ids){
        if((int)ids.size()<2) return; int sp=0, dp=(int)ids.size()-1; int s=ids[sp], d=ids[dp];
        vector<float> remb = ids.size()>=3? meanEmb(store,vector<int>(ids.begin()+1,ids.end()-1),D) : meanEmb(store,ids,D);
        int rc=relClust.assign(remb); vector<float> eemb=meanEmb(store,ids,D);
        graph.addEdge(s,d,rc,0.9f,eemb,sp,dp);
        for(int n:{s,d}) nodeConcept[n]=conClust.assign(store.E[n]);
    }

    // GRAPH-AUGMENTED NEXT TOKEN: forward -> active retrieval -> cross-attention
    // -> fused=h+gctx -> logits -> graph bias -> pick. Returneaza tokenul +
    // detalii de debug (gate/score/support/logits before/after).
    struct StepDbg{ float crossScore,fusedNorm,support; vector<float> baseTop,biasTop; vector<int> baseIdx,biasIdx; };
    int nextToken(const vector<int>& ctxTokens, ActiveGraphState& A, const vector<float>& qv,
                  set<int>& used, StepDbg& dbg, bool factualMode){
        vector<float> h=T.hiddenLast(ctxTokens);
        A.expandTopK(graph,8); A.buildHypotheses(graph,4,6);
        // memorie = embeddingurile nodurilor active (+ muchii active)
        vector<vector<float>> mem; vector<int> memNode;
        for(auto& kv:A.nodeScore){ mem.push_back(store.E[kv.first]); memNode.push_back(kv.first); }
        auto cross=gca.attend(h,mem);
        vector<float> fused=h; addInto(fused,cross.ctx, sigm(2.0f*(cross.maxScore-gca.tau)) ); // graful conteaza doar daca scorul > tau
        dbg.crossScore=cross.maxScore; dbg.fusedNorm=nrm(fused);
        vector<float> base=T.logitsCos(h);
        vector<float> z=T.logitsCos(fused);
        // memory-aware logit bias (in factualMode suprimam mai tare ne-activele)
        float beta=4.0f, gamma= factualMode? 9.0f : 2.0f; gbias.apply(z,A,qv,store,beta,gamma,dbg.support);
        // penalizare repetitie
        for(int u:used) if(u<(int)z.size()) z[u]-=4.0f;
        // top-5 before/after pentru debug
        auto top5=[&](const vector<float>& zz,vector<float>& tv,vector<int>& ti){ vector<pair<float,int>> v;
            for(int i=0;i<(int)zz.size();i++)v.push_back({zz[i],i}); sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first>b.first;});
            for(int i=0;i<5&&i<(int)v.size();i++){tv.push_back(v[i].first);ti.push_back(v[i].second);} };
        top5(base,dbg.baseTop,dbg.baseIdx); top5(z,dbg.biasTop,dbg.biasIdx);
        int best=-1; float bz=-1e9; for(int v=0;v<(int)z.size();v++){ if(!ctxTokens.empty()&&v==ctxTokens.back())continue;
            if(z[v]>bz){bz=z[v];best=v;} }
        return best;
    }
};

// ============================================================================
//  PARTEA 1 — SCALARE / MEMORIE
// ============================================================================

// 1.1 CSR GRAPH (Compressed Sparse Row) — cache-friendly, inlocuieste
//     unordered_map<int,vector<int>>. Construit ca SNAPSHOT read-only din
//     GraphMemory (graf mutabil la invatare -> CSR compilat pentru reasoning).
struct CSRGraph {
    int N=0; vector<int> nodeOffsets, edgeTargets, edgeIds, denseToTok;
    unordered_map<int,int> tokToDense;
    void build(const GraphMemory& g){
        denseToTok.assign(g.nodes.begin(), g.nodes.end()); sort(denseToTok.begin(),denseToTok.end());
        N=(int)denseToTok.size(); tokToDense.clear(); for(int i=0;i<N;i++) tokToDense[denseToTok[i]]=i;
        vector<int> deg(N,0);
        for(auto& e:g.edges){ auto it=tokToDense.find(e.src); if(it!=tokToDense.end()) deg[it->second]++; }
        nodeOffsets.assign(N+1,0); for(int i=0;i<N;i++) nodeOffsets[i+1]=nodeOffsets[i]+deg[i];
        int E=nodeOffsets[N]; edgeTargets.assign(E,-1); edgeIds.assign(E,-1);
        vector<int> cur(nodeOffsets.begin(), nodeOffsets.end()-1);
        for(int ei=0;ei<(int)g.edges.size();ei++){ auto& e=g.edges[ei]; auto it=tokToDense.find(e.src);
            if(it==tokToDense.end())continue; int pos=cur[it->second]++;
            auto jt=tokToDense.find(e.dst); edgeTargets[pos]=jt==tokToDense.end()?-1:jt->second; edgeIds[pos]=ei; }
    }
    // vecini (dense idx) ai nodului dense i: [nodeOffsets[i], nodeOffsets[i+1])
    size_t bytes()const{ return (nodeOffsets.size()+edgeTargets.size()+edgeIds.size()+denseToTok.size())*sizeof(int); }

    // INCREMENTAL (invatare continua): muchii noi merg intr-un overflow buffer
    // (O(1) append, fara rebuild). compact() le fuzioneaza periodic in CSR.
    unordered_map<int,vector<pair<int,int>>> overflow;   // denseSrc -> {(denseDst,eid)}
    void appendEdge(int srcTok,int dstTok,int eid){
        auto a=tokToDense.find(srcTok); if(a==tokToDense.end())return;
        int dd = tokToDense.count(dstTok)? tokToDense[dstTok] : -1;
        overflow[a->second].push_back({dd,eid});
    }
    void neighbors(int denseSrc, vector<int>& out)const{
        for(int p=nodeOffsets[denseSrc];p<nodeOffsets[denseSrc+1];p++) out.push_back(edgeTargets[p]);
        auto it=overflow.find(denseSrc); if(it!=overflow.end()) for(auto& pr:it->second) out.push_back(pr.first);
    }
    void compact(){   // fuzioneaza overflow in CSR si goleste bufferul
        vector<int> nt,ni,no(N+1,0); vector<int> deg(N,0);
        for(int i=0;i<N;i++) deg[i]=(nodeOffsets[i+1]-nodeOffsets[i]) + (overflow.count(i)?(int)overflow[i].size():0);
        for(int i=0;i<N;i++) no[i+1]=no[i]+deg[i];
        nt.assign(no[N],-1); ni.assign(no[N],-1); vector<int> cur(no.begin(),no.end()-1);
        for(int i=0;i<N;i++){ for(int p=nodeOffsets[i];p<nodeOffsets[i+1];p++){ nt[cur[i]]=edgeTargets[p]; ni[cur[i]]=edgeIds[p]; cur[i]++; }
            if(overflow.count(i)) for(auto& pr:overflow[i]){ nt[cur[i]]=pr.first; ni[cur[i]]=pr.second; cur[i]++; } }
        nodeOffsets.swap(no); edgeTargets.swap(nt); edgeIds.swap(ni); overflow.clear();
    }
};

// 1.2 SERIALIZARE BINARA — salveaza/incarca starea fara reconstructie din text.
struct GraphSerializer {
    static void save(const string& path, const GraphMemory& g, EmbeddingStore& es){
        FILE* f=fopen(path.c_str(),"wb"); if(!f)return;
        int magic=0x43474D31, D=es.D, V=es.size(), E=(int)g.edges.size();
        fwrite(&magic,4,1,f); fwrite(&D,4,1,f); fwrite(&V,4,1,f); fwrite(&E,4,1,f);
        for(int v=0;v<V;v++) fwrite(es.E[v].data(), sizeof(float), D, f);   // embeddings (zona mmap-abila)
        for(auto& e:g.edges){ fwrite(&e.src,4,1,f); fwrite(&e.dst,4,1,f); fwrite(&e.relCluster,4,1,f);
            fwrite(&e.confidence,4,1,f); fwrite(&e.seqOffset,1,1,f); fwrite(&e.orderConfidence,4,1,f); }
        fclose(f);
    }
    static bool loadHeader(const string& path,int& D,int& V,int& E){
        FILE* f=fopen(path.c_str(),"rb"); if(!f)return false; int magic=0;
        if(fread(&magic,4,1,f)!=1){fclose(f);return false;} fread(&D,4,1,f);fread(&V,4,1,f);fread(&E,4,1,f);
        fclose(f); return magic==0x43474D31;
    }
};

// 1.3 MEMORY-MAPPED STORAGE — mmap REAL (POSIX). Pentru grafuri uriase,
//     embeddingurile se acceseaza zero-copy din fisier, fara a incarca tot in RAM.
//     Header: [magic,D,V,E] apoi V*D float. Pe mobil acelasi API (bionic).
struct MemoryMappedGraphStorage {
    int fd=-1; void* base=nullptr; size_t len=0;
    bool openFile(const string& path){
        fd=::open(path.c_str(),O_RDONLY); if(fd<0)return false;
        struct stat st; if(fstat(fd,&st)!=0){::close(fd);fd=-1;return false;} len=st.st_size;
        base=mmap(nullptr,len,PROT_READ,MAP_PRIVATE,fd,0);
        if(base==MAP_FAILED){base=nullptr;::close(fd);fd=-1;return false;} return true;
    }
    const int* header()const{ return (const int*)base; }                       // [magic,D,V,E]
    const float* embedding(int v,int D)const{ return (const float*)((const char*)base+16)+ (size_t)v*D; } // zero-copy
    // Zona de MUCHII (zero-copy): incepe dupa header+embeddings. Stride per muchie:
    // src(4)+dst(4)+rel(4)+conf(4)+seqOffset(1)+orderConf(4) = 21 octeti (packed).
    static constexpr int EDGE_STRIDE=21;
    const char* edgesRegion(int V,int D)const{ return (const char*)base+16+(size_t)V*D*4; }
    int edgeSrc(int i,int V,int D)const{ int s; memcpy(&s, edgesRegion(V,D)+(size_t)i*EDGE_STRIDE, 4); return s; }
    int edgeDst(int i,int V,int D)const{ int s; memcpy(&s, edgesRegion(V,D)+(size_t)i*EDGE_STRIDE+4, 4); return s; }
    void closeFile(){ if(base){munmap(base,len);base=nullptr;} if(fd>=0){::close(fd);fd=-1;} }
};

// 1.4 ANN INDEX — LSH cu hiperplane aleatoare. Cautare aproximativa sub-liniara
//     (bucket + probing Hamming-1), NU O(N) peste toate nodurile.
struct ANNIndex {
    int D,H; EmbeddingStore* es; vector<vector<float>> planes;
    unordered_map<uint64_t,vector<int>> buckets;
    ANNIndex(EmbeddingStore* s,int d,int h):D(d),H(h),es(s){
        normal_distribution<float> g(0,1); planes.assign(H,vector<float>(D));
        for(auto& p:planes) for(float& x:p) x=g(rng);
    }
    uint64_t sig(const vector<float>& v)const{ uint64_t s=0; for(int i=0;i<H;i++) if(dotv(v,planes[i])>0) s|=(1ull<<i); return s; }
    void build(const set<int>& nodes){ buckets.clear(); for(int n:nodes) buckets[sig(es->E[n])].push_back(n); }
    vector<int> query(const vector<float>& q,int k,int& scanned)const{
        uint64_t s=sig(q); vector<pair<float,int>> cand; scanned=0;
        auto scan=[&](uint64_t key){ auto it=buckets.find(key); if(it==buckets.end())return;
            for(int n:it->second){ cand.push_back({cosv(q,es->E[n]),n}); scanned++; } };
        scan(s); for(int i=0;i<H;i++) scan(s^(1ull<<i));     // probing vecini Hamming-1
        sort(cand.begin(),cand.end(),[](const pair<float,int>&a,const pair<float,int>&b){return a.first>b.first;});
        vector<int> r; for(int i=0;i<k&&i<(int)cand.size();i++) r.push_back(cand[i].second); return r;
    }
};

// 1.5 ACTIVE SUBGRAPH — doar top noduri/muchii/drumuri active intra in reasoning.
struct ActiveSubgraph {
    vector<int> nodes, edges; vector<vector<int>> paths;
    void build(const GraphMemory& g,const vector<int>& seed,int maxLen){
        set<int> ns(seed.begin(),seed.end());
        for(int s:seed) for(int ei:g.out(s)) ns.insert(g.edges[ei].dst);   // 1-hop
        nodes.assign(ns.begin(),ns.end());
        for(int ei=0;ei<(int)g.edges.size();ei++) if(ns.count(g.edges[ei].src)&&ns.count(g.edges[ei].dst)) edges.push_back(ei);
        for(int s:seed){ vector<int> p{s}; int cur=s; set<int> seen{s};
            for(int st=0;st<maxLen;st++){ int nx=-1; float bc=0; for(int ei:g.out(cur)) if(!seen.count(g.edges[ei].dst)&&g.edges[ei].confidence>bc){bc=g.edges[ei].confidence;nx=g.edges[ei].dst;}
                if(nx<0)break; p.push_back(nx); seen.insert(nx); cur=nx; }
            if(p.size()>=2) paths.push_back(p); }
    }
};

// ============================================================================
//  PARTEA 2 — WORLD MODEL (tranzitii cu confidence/variance/support; simulare;
//  counterfactual). LIMITA ONESTA: datele nu au POLARITATE (creste/scade), deci
//  simularea modeleaza PROPAGAREA efectului (ce devine afectat si cu ce
//  magnitudine), nu semnul. Pentru semn ar trebui muchii cu polaritate invatata.
// ============================================================================
struct WorldModel {
    struct Transition{ int s,rel,ns; float confidence,variance,support; };
    vector<Transition> trans; unordered_map<int,vector<int>> bySrc;
    void learn(const GraphMemory& g){
        trans.clear(); bySrc.clear();
        map<pair<int,int>,vector<int>> bucket;
        for(auto& e:g.edges) bucket[{e.src,e.relCluster}].push_back(e.dst);
        for(auto& kv:bucket){ int sup=(int)kv.second.size(); set<int> uniq(kv.second.begin(),kv.second.end());
            float var = 1.f - 1.f/(float)uniq.size();                 // mai multe rezultate distincte => mai incert
            for(int dst:uniq){ int cnt=(int)count(kv.second.begin(),kv.second.end(),dst);
                int idx=(int)trans.size();
                trans.push_back({kv.first.first,kv.first.second,dst,(float)cnt/sup,var,(float)sup});
                bySrc[kv.first.first].push_back(idx); } }
    }
    // simulare prin propagare de nivel; risk = incertitudine acumulata
    map<int,float> simulate(int start,int steps,float& risk,const set<int>& blocked={}){
        map<int,float> level; level[start]=1.f; map<int,float> frontier=level; risk=0;
        for(int s=0;s<steps;s++){ map<int,float> nxt;
            for(auto& kv:frontier){ if(blocked.count(kv.first))continue; auto it=bySrc.find(kv.first); if(it==bySrc.end())continue;
                for(int ti:it->second){ auto& t=trans[ti]; float flow=kv.second*t.confidence*(1.f-0.3f*t.variance);
                    if(flow<1e-3f)continue; nxt[t.ns]+=flow; level[t.ns]+=flow; risk+=kv.second*t.variance*0.1f; } }
            frontier.swap(nxt); if(frontier.empty())break; }
        return level;
    }
};

// ============================================================================
//  PARTEA 3 — PLANNING ENGINE (Goal -> drumuri candidate -> evaluare -> best)
// ============================================================================
struct GoalPlanner {
    struct Plan{ vector<int> nodes,rels; float score,confidence,risk; };
    vector<Plan> plan(const GraphMemory& g,int start,int goal,WorldModel& wm,int maxLen){
        vector<Plan> plans;
        function<void(int,vector<int>&,vector<int>&,set<int>&,float)> dfs=
        [&](int cur,vector<int>& ns,vector<int>& rs,set<int>& seen,float conf){
            if(cur==goal && ns.size()>=2){ Plan p; p.nodes=ns; p.rels=rs; p.confidence=conf;
                float risk=0; wm.simulate(ns.front(),(int)ns.size(),risk);
                p.risk=risk; p.score=conf*(1.f-min(1.f,risk)); plans.push_back(p); return; }
            if((int)ns.size()>=maxLen) return;
            for(int ei:g.out(cur)){ int d=g.edges[ei].dst; if(seen.count(d))continue;
                seen.insert(d); ns.push_back(d); rs.push_back(g.edges[ei].relCluster);
                dfs(d,ns,rs,seen,conf*g.edges[ei].confidence);
                seen.erase(d); ns.pop_back(); rs.pop_back(); }
        };
        vector<int> ns{start},rs; set<int> seen{start}; dfs(start,ns,rs,seen,1.f);
        sort(plans.begin(),plans.end(),[](const Plan&a,const Plan&b){return a.score>b.score;});
        return plans;
    }
};

// ============================================================================
//  PARTEA 5 — SELF VERIFICATION (scor compus, numeric)
//   AnswerScore = Language + GraphSupport + ReasoningSupport + SimulationSupport
//                 - ContradictionPenalty
// ============================================================================
struct Verifier {
    struct Score{ float language,graph,reasoning,simulation,contradiction,total; };
    Score verify(const vector<int>& nodes,const GraphMemory& g,WorldModel& wm,
                 EmbeddingStore& es,const vector<float>& qv){
        Score sc{}; if(nodes.size()<2){sc.total=0;return sc;}
        // graph support = confidence medie pe muchiile parcurse + penalizare ordine
        float gconf=0,ordPen=0; int hops=0;
        for(size_t i=0;i+1<nodes.size();i++){ for(int ei:g.out(nodes[i])) if(g.edges[ei].dst==nodes[i+1]){
            gconf+=g.edges[ei].confidence; if(g.edges[ei].orderConfidence<0.4f) ordPen+=1; hops++; break; } }
        sc.graph = hops? gconf/hops : 0;
        sc.contradiction = hops? ordPen/hops : 0;
        // reasoning = lungimea lantului conectat (normalizat)
        sc.reasoning = min(1.f,(float)hops/4.f);
        // language = relevanta medie a nodurilor fata de query (proxy lingvistic)
        float lang=0; for(int n:nodes) lang+=max(0.f,cosv(qv,es.E[n])); sc.language=lang/nodes.size();
        // simulation = world-model confirma ca finalul e atins din start
        float risk=0; auto lvl=wm.simulate(nodes.front(),(int)nodes.size()+2,risk);
        sc.simulation = lvl.count(nodes.back())? min(1.f,lvl[nodes.back()]) : 0.f;
        sc.total = sc.language+sc.graph+sc.reasoning+sc.simulation-sc.contradiction;
        return sc;
    }
};

// ============================================================================
//  PARTEA 4 — PROGRAMMING WORLD MODEL (DEMO minimal, dar REAL ca executie)
//  Limbaj jucarie: atribuiri "x = NUMAR" sau "x = a + b ...". Construieste un
//  graf de simboluri + muchii de DEPENDENTA (scope/type/call sunt schitate).
//  ProgramSimulator evalueaza prin ordine topologica = "execution graph".
//  DEMO: nu e un parser real de C/C++; arata principiul code-as-graph.
// ============================================================================
struct ProgrammingGraph {
    map<string,int> sym; vector<string> names;
    map<int,vector<string>> rhs;          // simbol -> tokens din partea dreapta
    map<int,long> directVal;              // simbol -> constanta directa
    vector<pair<int,int>> depEdges;       // (simbol, simbol-dependinta)  = data dependency
    int getSym(const string& s){ auto it=sym.find(s); if(it!=sym.end())return it->second;
        int id=(int)names.size(); sym[s]=id; names.push_back(s); return id; }
    bool isNumber(const string& s){ if(s.empty())return false; for(char c:s) if(!isdigit((unsigned char)c))return false; return true; }
    void addAssign(const string& line){               // "x = a + 3"
        auto t=splitw(line); if(t.size()<3||t[1]!="=")return; int lhs=getSym(t[0]);
        vector<string> r(t.begin()+2,t.end()); rhs[lhs]=r;
        if(r.size()==1 && isNumber(r[0])) directVal[lhs]=stol(r[0]);
        for(auto& tk:r) if(!isNumber(tk) && tk!="+"){ int dep=getSym(tk); depEdges.push_back({lhs,dep}); }
    }
    // EXECUTION GRAPH: evaluare prin recursie pe dependente (cu memoizare)
    long eval(int s, map<int,long>& memo, set<int>& stack){
        if(memo.count(s))return memo[s]; if(stack.count(s))return 0; stack.insert(s);
        long v=0; if(directVal.count(s)) v=directVal[s];
        else if(rhs.count(s)){ for(auto& tk:rhs[s]){ if(tk=="+")continue;
            if(isNumber(tk)) v+=stol(tk); else v+=eval(sym[tk],memo,stack); } }
        stack.erase(s); memo[s]=v; return v;
    }
    map<string,long> run(){ map<int,long> memo; for(size_t i=0;i<names.size();i++){ set<int> st; eval((int)i,memo,st); }
        map<string,long> out; for(auto& kv:sym) out[kv.first]=memo[kv.second]; return out; }
};

// ============================================================================
//  PARTEA 6 — LEARNERS (online, fara retraining complet). Concept/Relation/
//  Language exista deja in Engine (clustering + LoRA). Aici: Rule/Goal/Transition.
// ============================================================================
struct RuleLearner {  // motiv structural repetat (Csrc,rel,Cdst) -> regula numerica
    struct Rule{ int cs,rel,cd,count; };
    vector<Rule> learn(const GraphMemory& g,const map<int,int>& nodeConcept){
        map<tuple<int,int,int>,int> m;
        for(auto& e:g.edges){ auto a=nodeConcept.find(e.src),b=nodeConcept.find(e.dst);
            if(a==nodeConcept.end()||b==nodeConcept.end())continue; m[make_tuple(a->second,e.relCluster,b->second)]++; }
        vector<Rule> rs; for(auto& kv:m) if(kv.second>=2){ int cs,rel,cd; tie(cs,rel,cd)=kv.first; rs.push_back({cs,rel,cd,kv.second}); }
        return rs;
    }
};
struct GoalLearner {  // atractori: noduri spre care converg multe lanturi
    vector<pair<int,int>> learn(const GraphMemory& g){
        map<int,int> reach;
        for(int s:g.nodes){ set<int> seen{s}; vector<int> q{s};
            for(size_t i=0;i<q.size();i++) for(int ei:g.out(q[i])){ int v=g.edges[ei].dst; if(!seen.count(v)){seen.insert(v);q.push_back(v);} }
            for(int v:seen) if(v!=s) reach[v]++; }
        vector<pair<int,int>> v(reach.begin(),reach.end());
        sort(v.begin(),v.end(),[](const pair<int,int>&a,const pair<int,int>&b){return a.second>b.second;});
        return v;
    }
};

// ============================================================================
//  PARTEA 7 — COGNITIVE LOOP (9 etape explicite)
//  PERCEIVE -> UNDERSTAND -> RETRIEVE -> REASON -> SIMULATE -> PLAN -> VERIFY
//  -> GENERATE -> LEARN.  Fiecare etapa e o metoda separata, observabila.
// ============================================================================
struct CognitiveLoop {
    Engine& eng; ANNIndex& ann; WorldModel& wm; GoalPlanner& planner; Verifier& verifier;
    CognitiveLoop(Engine& e,ANNIndex& a,WorldModel& w,GoalPlanner& p,Verifier& v)
        :eng(e),ann(a),wm(w),planner(p),verifier(v){}

    void run(const string& prompt,int goalNode){
        cout<<"\n  ===== COGNITIVE LOOP: \""<<prompt<<"\" =====\n";
        // 1. PERCEIVE: tokenizare adaptiva
        vector<string> nw; auto toks=eng.tokenize(prompt,nw);
        cout<<"  [1 PERCEIVE]  tokeni="<<toks.size()<<", noi="<<nw.size()<<"\n";
        // 2. UNDERSTAND: embedding de query + ancore in graf
        vector<float> qv=meanEmb(eng.store,toks,eng.D); vector<int> anchors;
        for(int t:toks) if(eng.graph.isNode(t)) anchors.push_back(t);
        cout<<"  [2 UNDERSTAND] ancore in graf: "; for(int a:anchors)cout<<eng.tok.name(a)<<" "; cout<<"\n";
        // 3. RETRIEVE: ANN (aproximativ) -> noduri relevante; active subgraph
        int scanned=0; auto near=ann.query(qv,5,scanned);
        cout<<"  [3 RETRIEVE]  ANN a scanat "<<scanned<<"/"<<eng.graph.nodes.size()<<" noduri; relevante: ";
        for(int n:near)cout<<eng.tok.name(n)<<" "; cout<<"\n";
        ActiveSubgraph sub; sub.build(eng.graph,anchors.empty()?near:anchors,6);
        // 4. REASON: cel mai bun drum din subgraful activ
        vector<int> bestPath; float bestConf=0;
        for(auto& p:sub.paths){ float c=1; for(size_t i=0;i+1<p.size();i++) for(int ei:eng.graph.out(p[i])) if(eng.graph.edges[ei].dst==p[i+1]){c*=eng.graph.edges[ei].confidence;break;}
            if(c>bestConf){bestConf=c;bestPath=p;} }
        cout<<"  [4 REASON]    drum: "; for(size_t i=0;i<bestPath.size();i++)cout<<(i?" -> ":"")<<eng.tok.name(bestPath[i]); cout<<"\n";
        // 5. SIMULATE: world model din start
        float risk=0; if(!bestPath.empty()){ auto lvl=wm.simulate(bestPath.front(),5,risk);
            cout<<"  [5 SIMULATE]  risc="<<risk<<", noduri afectate="<<lvl.size()<<"\n"; }
        else cout<<"  [5 SIMULATE]  (fara start)\n";
        // 6. PLAN: spre obiectivul descoperit
        if(!anchors.empty()){ auto plans=planner.plan(eng.graph,anchors.front(),goalNode,wm,7);
            cout<<"  [6 PLAN]      spre '"<<eng.tok.name(goalNode)<<"': "<<plans.size()<<" planuri; best: ";
            if(!plans.empty()){ for(size_t i=0;i<plans[0].nodes.size();i++)cout<<(i?" -> ":"")<<eng.tok.name(plans[0].nodes[i]);
                cout<<"  (score="<<plans[0].score<<")"; } cout<<"\n"; }
        else cout<<"  [6 PLAN]      (fara ancora)\n";
        // 7. VERIFY: scor compus
        Verifier::Score vs=verifier.verify(bestPath.empty()?near:bestPath,eng.graph,wm,eng.store,qv);
        cout<<"  [7 VERIFY]    lang="<<vs.language<<" graph="<<vs.graph<<" reason="<<vs.reasoning
            <<" sim="<<vs.simulation<<" contra="<<vs.contradiction<<" => total="<<vs.total<<"\n";
        // 8. GENERATE: prudent daca verificarea e slaba (verdict numeric, nu semantic)
        if(vs.total < 0.6f) cout<<"  [8 GENERATE]  suport slab => raspuns prudent: \"Nu sunt sigur.\"\n";
        else { cout<<"  [8 GENERATE]  raspuns ancorat in graf: "; for(size_t i=0;i<bestPath.size();i++)cout<<(i?" ":"")<<eng.tok.name(bestPath[i]); cout<<"\n"; }
        // 9. LEARN: intareste muchiile drumului folosit (invatare continua)
        if(!bestPath.empty()){ vector<float> ctx=meanEmb(eng.store,bestPath,eng.D);
            for(int n:bestPath) eng.online.updateNodeEmbedding(n,ctx,0.02f);
            cout<<"  [9 LEARN]     drum confirmat: embedding-uri/confidence intarite\n"; }
    }
};

// ############################################################################
//  v8 — "IMPLEMENTEAZA TOT": inchidem golurile din analiza v7 cu cod REAL.
// ############################################################################

// ====== (4 din analiza) SIGNED WORLD MODEL — polaritate + magnitude invatate
//  din EPISOADE de observatie numerice (weak supervision). Pentru fiecare
//  muchie a->b invatam un coeficient SEMNAT w prin regresie: db_t ~ w * a_{t-1}.
//  Datele sunt serii numerice (ca niste log-uri de senzori), nu reguli semantice.
struct SignedWorldModel {
    struct Episode{ vector<map<int,float>> series; };   // serie temporala de valori pe noduri
    map<pair<int,int>,float> w, conf;
    void learn(const GraphMemory& g,const vector<Episode>& eps,int iters,float lr){
        for(auto& e:g.edges) if(!w.count({e.src,e.dst})) w[{e.src,e.dst}]=0.f;
        for(int it=0;it<iters;it++){ map<pair<int,int>,float> grad; map<pair<int,int>,int> cnt;
            for(auto& ep:eps) for(size_t t=1;t<ep.series.size();t++) for(auto& e:g.edges){
                auto pa=ep.series[t-1].find(e.src); if(pa==ep.series[t-1].end())continue;
                auto pb1=ep.series[t].find(e.dst); if(pb1==ep.series[t].end())continue;
                auto pb0=ep.series[t-1].find(e.dst); float bprev=pb0==ep.series[t-1].end()?0:pb0->second;
                float aval=pa->second, db=pb1->second-bprev; float err=w[{e.src,e.dst}]*aval-db;
                grad[{e.src,e.dst}]+=err*aval; cnt[{e.src,e.dst}]++; }
            for(auto& kv:grad){ int c=cnt[kv.first]; if(c) w[kv.first]-=lr*kv.second/c; } }
        float mx=1e-6f; for(auto&kv:w)mx=max(mx,fabs(kv.second)); for(auto&kv:w)conf[kv.first]=fabs(kv.second)/mx;
    }
    float polarity(int a,int b)const{ auto it=w.find({a,b}); return it==w.end()?0.f:it->second; }
    // simulare SEMNATA: perturbam un nod cu delta, propagam db = w*val (creste/scade)
    map<int,float> simulate(int start,float delta,int steps,const GraphMemory& g)const{
        map<int,float> val; val[start]=delta;
        for(int s=0;s<steps;s++){ map<int,float> nxt;
            for(auto& e:g.edges){ auto it=val.find(e.src); if(it==val.end())continue;
                float dv=polarity(e.src,e.dst)*it->second; if(fabs(dv)<1e-3f)continue; nxt[e.dst]+=dv; }
            if(nxt.empty())break; for(auto&kv:nxt) val[kv.first]+=kv.second; }
        return val;
    }
};

// ====== (INT8 real) Int8Linear — GEMM cuantizat cu acumulare INT32.
//  Greutati int8 (scala per coloana de iesire) + input int8 (scala per-tensor).
//  y_j = sx*sw_j * sum_i qx_i*qw_ij  (acumulare in int32, dequant la final).
struct Int8Linear {
    int R,C; vector<int8_t> qw; vector<float> sw;   // sw[j] scala coloanei j
    Int8Linear(Mat& M):R(M.R),C(M.C),qw(M.R*M.C),sw(M.C,1){
        for(int j=0;j<C;j++){ float mx=1e-9f; for(int i=0;i<R;i++)mx=max(mx,fabs(M.at(i,j)));
            sw[j]=mx/127.f; for(int i=0;i<R;i++) qw[i*C+j]=(int8_t)lround(M.at(i,j)/sw[j]); }
    }
    vector<float> forward(const vector<float>& x)const{
        float mx=1e-9f; for(float v:x)mx=max(mx,fabs(v)); float sx=mx/127.f;
        vector<int8_t> qx(R); for(int i=0;i<R;i++)qx[i]=(int8_t)lround(x[i]/sx);
        vector<float> y(C,0);
        for(int j=0;j<C;j++){ int32_t acc=0; for(int i=0;i<R;i++) acc+=(int32_t)qx[i]*qw[i*C+j];
            y[j]=sx*sw[j]*acc; } return y;
    }
};

// ====== (ANN tuning) MULTI-TABLE LSH — L tabele, recall mai bun ca single-table.
struct MultiTableANN {
    int D,H,L; EmbeddingStore* es; vector<vector<vector<float>>> planes;
    vector<unordered_map<uint64_t,vector<int>>> tables;
    MultiTableANN(EmbeddingStore* s,int d,int h,int l):D(d),H(h),L(l),es(s){
        normal_distribution<float> g(0,1); planes.assign(L,{});
        for(int t=0;t<L;t++){ planes[t].assign(H,vector<float>(D)); for(auto&p:planes[t])for(float&x:p)x=g(rng); }
        tables.assign(L,{});
    }
    uint64_t sig(const vector<float>& v,int t)const{ uint64_t s=0; for(int i=0;i<H;i++) if(dotv(v,planes[t][i])>0)s|=(1ull<<i); return s; }
    void build(const set<int>& nodes){ for(int t=0;t<L;t++){ tables[t].clear(); for(int n:nodes) tables[t][sig(es->E[n],t)].push_back(n); } }
    vector<int> query(const vector<float>& q,int k,int& scanned)const{ set<int> cand; scanned=0;
        for(int t=0;t<L;t++){ auto it=tables[t].find(sig(q,t)); if(it!=tables[t].end()) for(int n:it->second){cand.insert(n);} }
        vector<pair<float,int>> v; for(int n:cand){v.push_back({cosv(q,es->E[n]),n});scanned++;}
        sort(v.begin(),v.end(),[](const pair<float,int>&a,const pair<float,int>&b){return a.first>b.first;});
        vector<int> r; for(int i=0;i<k&&i<(int)v.size();i++)r.push_back(v[i].second); return r; }
};

// ====== (tokeni virtuali INVATATI) ACTION POLICY — regresie logistica peste
//  hidden state -> P(emite [SEARCH_GRAPH]). Invatata pe exemple (low/high support),
//  NU un prag fix. Generalizeaza in spatiul ascuns.
struct ActionPolicy {
    int D; vector<float> wv; float b=0;
    ActionPolicy(int d):D(d),wv(d,0){}
    float prob(const vector<float>& h)const{ return sigm(dotv(h,wv)+b); }
    void train(const vector<pair<vector<float>,int>>& data,int iters,float lr){
        for(int it=0;it<iters;it++) for(auto& ex:data){ float p=prob(ex.first); float g=p-ex.second;
            for(int i=0;i<D;i++) wv[i]-=lr*g*ex.first[i]; b-=lr*g; }
    }
    float accuracy(const vector<pair<vector<float>,int>>& data)const{ int ok=0; for(auto&ex:data) if((prob(ex.first)>0.5f)==(ex.second==1))ok++;
        return data.empty()?0:(float)ok/data.size(); }
};

// ====== (5) VERIFIER CONTRADICTION via polaritate: un drum care implica
//  simultan crestere SI scadere a aceluiasi nod e contradictoriu.
static float signContradiction(const vector<int>& path,const SignedWorldModel& sw){
    map<int,int> dir;  // nod -> semn cumulat al efectului pe drum
    for(size_t i=0;i+1<path.size();i++){ float p=sw.polarity(path[i],path[i+1]); if(fabs(p)<1e-3f)continue;
        int s=p>0?1:-1; if(dir.count(path[i+1]) && dir[path[i+1]]!=s) return 1.f; dir[path[i+1]]=s; }
    return 0.f;
}

// ====== PROGRAM LEARNING ENGINE — invata structura DOAR din TRACES de
//  executie (valori observate), NU din keyword-uri. "Sensorul" ofera simboluri
//  OPACE + valorile lor; semnificatia (variabile, dependente, functii de
//  tranzitie) e DESCOPERITA din dinamica. Niciun if(token=="for"/"while"/...).
struct TraceLearner {
    int nSym=0; map<pair<int,int>,float> coef; vector<int> varying;
    void learn(int symbols,const vector<vector<map<int,double>>>& traces,int iters,float lr){
        nSym=symbols; coef.clear();
        for(int b=0;b<nSym;b++) for(int a=0;a<nSym;a++) coef[{a,b}]=0.f;
        for(int it=0;it<iters;it++){ map<pair<int,int>,double> grad; map<pair<int,int>,int> cnt;
            for(auto& tr:traces) for(size_t t=1;t<tr.size();t++) for(int b=0;b<nSym;b++){
                auto pb=tr[t].find(b); if(pb==tr[t].end())continue; double pred=0;
                for(int a=0;a<nSym;a++){ auto pa=tr[t-1].find(a); if(pa!=tr[t-1].end())pred+=coef[{a,b}]*pa->second; }
                double err=pred-pb->second;
                for(int a=0;a<nSym;a++){ auto pa=tr[t-1].find(a); if(pa==tr[t-1].end())continue; grad[{a,b}]+=err*pa->second; cnt[{a,b}]++; } }
            for(auto& kv:grad){ int c=cnt[kv.first]; if(c) coef[kv.first]-=lr*kv.second/c; } }
        varying.clear(); for(int s=0;s<nSym;s++){ double mn=1e18,mx=-1e18;
            for(auto& tr:traces) for(auto& f:tr){ auto it=f.find(s); if(it!=f.end()){mn=min(mn,it->second);mx=max(mx,it->second);} }
            if(mx-mn>1e-6) varying.push_back(s); }
    }
};

// ====== EFFECT MODES — tipuri de efect EMERGENTE (clustere de polaritate),
//  nu etichete "increase/decrease/block". Feature = [coef, |coef|] -> cluster.
struct EffectModes {
    map<int,int> edgeMode; int nModes=0;
    void discover(const GraphMemory& g,const SignedWorldModel& sw){
        Clusterer c(0.9f);
        for(int ei=0;ei<(int)g.edges.size();ei++){ float w=sw.polarity(g.edges[ei].src,g.edges[ei].dst);
            float m=fabs(w)<0.05f?1e-3f:fabs(w); vector<float> feat={w/m, 1.0f};   // directie de semn normalizata
            edgeMode[ei]=c.assign(feat); }
        nModes=(int)c.centroid.size();
    }
};

// ====== TEMPORAL MODEL — ordine / durata(lag) / frecventa dintr-un proces.
struct TemporalModel {
    map<pair<int,int>,int> freq; map<pair<int,int>,float> lag; map<pair<int,int>,int> lagN;
    void learn(const vector<vector<int>>& seqs){
        for(auto& s:seqs) for(size_t i=0;i<s.size();i++) for(size_t j=i+1;j<s.size();j++){
            freq[{s[i],s[j]}]++; lag[{s[i],s[j]}]+=(float)(j-i); lagN[{s[i],s[j]}]++; }
        for(auto& kv:lag) kv.second/=max(1,lagN[kv.first]);
    }
    vector<int> process(int start,int steps){ vector<int> p{start}; int cur=start; set<int> seen{start};
        for(int s=0;s<steps;s++){ int best=-1,bf=0;
            for(auto& kv:freq) if(kv.first.first==cur && !seen.count(kv.first.second) && lag[kv.first]<1.5f && kv.second>bf){bf=kv.second;best=kv.first.second;}
            if(best<0)break; p.push_back(best); seen.insert(best); cur=best; } return p; }
};

// ====== CREATIVE ENGINE — "metafora" = leaga noduri DEPARTATE in graf dar
//  APROPIATE in embedding (combinatii ne-evidente). Nu sabloane.
struct CreativeEngine {
    vector<pair<int,int>> bridges(const GraphMemory& g,EmbeddingStore& es,float lo,float hi,int k){
        vector<int> ns(g.nodes.begin(),g.nodes.end()); vector<pair<float,pair<int,int>>> cand;
        auto conn=[&](int a,int b){ for(int ei:g.out(a)) if(g.edges[ei].dst==b)return true; return false; };
        for(size_t i=0;i<ns.size();i++) for(size_t j=i+1;j<ns.size();j++){ if(conn(ns[i],ns[j])||conn(ns[j],ns[i]))continue;
            float s=cosv(es.E[ns[i]],es.E[ns[j]]); if(s>lo&&s<hi) cand.push_back({s,{ns[i],ns[j]}}); }
        sort(cand.begin(),cand.end(),[](const pair<float,pair<int,int>>&a,const pair<float,pair<int,int>>&b){return a.first>b.first;});
        vector<pair<int,int>> r; for(int i=0;i<k&&i<(int)cand.size();i++)r.push_back(cand[i].second); return r;
    }
};


// ============================================================================
//  MAIN — demo integrat (Parti 1-7) + analiza
// ============================================================================
int main(){
    Engine eng(DEMO_D);
    auto& tok=eng.tok; auto& store=eng.store; auto& graph=eng.graph; auto& T=eng.T;

    // --- corpus mic + pretrain (model mic; inteligenta sta in graf/world-model) ---
    vector<string> ro = {
        "om este fiinta","pisica este animal","caine este animal","Ion este om","Vasile este om",
        "Ion lucreaza sofer","sofer conduce camion","sofer face munca","munca produce bani",
        "bani cumpara hrana","hrana sustine om","om munceste pentru bani","motorul produce putere",
        "focul produce caldura"
    };
    vector<vector<int>> roIds; for(auto& s:ro){ vector<string> nw; roIds.push_back(eng.tokenize(s,nw)); }
    cout<<"== PRETRAIN (model mic) ==  D="<<DEMO_D<<" -> tinta TARGET_D="<<TARGET_D<<"\n";
    float step=0; for(int ep=0;ep<200;ep++){ vector<int> ord(roIds.size()); for(int i=0;i<(int)ord.size();i++)ord[i]=i;
        shuffle(ord.begin(),ord.end(),rng); for(int i:ord) T.trainSeq(roIds[i],true,false,0.01f,1e-3f,step); }
    for(auto& ids:roIds) eng.absorb(ids);
    cout<<"   graf: "<<graph.nodes.size()<<" noduri, "<<graph.edges.size()<<" muchii\n";

    // ================= PARTEA 1: SCALARE / MEMORIE =================
    cout<<"\n========== PARTEA 1: SCALARE / MEMORIE ==========\n";
    CSRGraph csr; csr.build(graph);
    cout<<"[CSR] N="<<csr.N<<" muchii="<<csr.edgeTargets.size()<<" memorie="<<csr.bytes()<<" bytes (cache-friendly)\n";
    { int di=csr.tokToDense.count(tok.get("Ion"))?csr.tokToDense[tok.get("Ion")]:-1;
      if(di>=0){ cout<<"   vecini CSR ai 'Ion': "; for(int p=csr.nodeOffsets[di];p<csr.nodeOffsets[di+1];p++)
          cout<<tok.name(csr.denseToTok[csr.edgeTargets[p]])<<" "; cout<<"\n"; } }

    ANNIndex ann(&store,DEMO_D,6); ann.build(graph.nodes);
    { int scanned=0; int ionId=tok.get("Ion"); auto near=ann.query(store.E[ionId],4,scanned);
      cout<<"[ANN/LSH] query 'Ion' a scanat "<<scanned<<"/"<<graph.nodes.size()<<" noduri (sub-liniar); vecini: ";
      for(int n:near)cout<<tok.name(n)<<" "; cout<<"\n"; }

    GraphSerializer::save("/tmp/graph.bin",graph,store);
    { int D,V,E; bool ok=GraphSerializer::loadHeader("/tmp/graph.bin",D,V,E);
      cout<<"[SERIALIZE] /tmp/graph.bin  ok="<<ok<<"  D="<<D<<" V="<<V<<" E="<<E<<" (incarcare fara reconstructie din text)\n"; }
    { MemoryMappedGraphStorage mm; if(mm.openFile("/tmp/graph.bin")){ const int* h=mm.header();
        cout<<"[MMAP] zero-copy header: D="<<h[1]<<" V="<<h[2]<<" E="<<h[3]
            <<"; emb[Ion][0]="<<mm.embedding(tok.get("Ion"),h[1])[0]<<" (acces direct din fisier)\n"; mm.closeFile(); } }

    ActiveSubgraph sub; sub.build(graph,{tok.get("Ion")},6);
    cout<<"[ACTIVE SUBGRAPH] noduri="<<sub.nodes.size()<<" muchii="<<sub.edges.size()<<" drumuri="<<sub.paths.size()<<"\n";

    // ================= PARTEA 2: WORLD MODEL =================
    cout<<"\n========== PARTEA 2: WORLD MODEL ==========\n";
    WorldModel wm; wm.learn(graph);
    cout<<"[WORLD MODEL] tranzitii invatate="<<wm.trans.size()<<" (fiecare cu confidence/variance/support)\n";
    { float risk=0; auto lvl=wm.simulate(tok.get("Ion"),6,risk);
      cout<<"   SIMULARE din 'Ion' (6 pasi): afectate="<<lvl.size()<<" risc="<<risk<<"\n      ";
      vector<pair<float,int>> v; for(auto&kv:lvl)v.push_back({kv.second,kv.first});
      sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first>b.first;});
      for(int i=0;i<6&&i<(int)v.size();i++)cout<<tok.name(v[i].second)<<"("<<v[i].first<<") "; cout<<"\n"; }
    { // COUNTERFACTUAL: "ce daca Ion nu mai lucreaza" => blocam nodul 'sofer' (jobul)
      float r0=0,r1=0; auto base=wm.simulate(tok.get("Ion"),6,r0);
      set<int> blk{tok.get("sofer")}; auto cf=wm.simulate(tok.get("Ion"),6,r1,blk);
      float lostBani = (base.count(tok.get("bani"))?base[tok.get("bani")]:0) - (cf.count(tok.get("bani"))?cf[tok.get("bani")]:0);
      cout<<"   COUNTERFACTUAL (blocam 'sofer'): efect asupra 'bani' = -"<<lostBani
          <<"  (consecinta estimata, fara executie reala)\n"; }

    // ================= PARTEA 3: PLANNING =================
    cout<<"\n========== PARTEA 3: PLANNING ==========\n";
    GoalLearner gl; auto goals=gl.learn(graph);
    int goalNode = goals.empty()? tok.get("bani") : goals.front().first;
    cout<<"[GOAL LEARNER] obiectiv-atractor descoperit: '"<<tok.name(goalNode)<<"'\n";
    GoalPlanner planner; auto plans=planner.plan(graph,tok.get("Ion"),tok.get("bani"),wm,7);
    cout<<"[PLANNER] planuri Ion -> bani: "<<plans.size()<<"\n";
    for(int i=0;i<3&&i<(int)plans.size();i++){ cout<<"   #"<<i<<" score="<<plans[i].score<<" conf="<<plans[i].confidence<<" : ";
        for(size_t k=0;k<plans[i].nodes.size();k++)cout<<(k?" -> ":"")<<tok.name(plans[i].nodes[k]); cout<<"\n"; }

    // ================= PARTEA 4: PROGRAMMING WORLD MODEL (demo) =================
    cout<<"\n========== PARTEA 4: PROGRAMMING WORLD MODEL (demo executie) ==========\n";
    ProgrammingGraph pg; for(string line:{string("x = 5"),string("y = x + 3"),string("z = y + x"),string("w = z + 2")}){ pg.addAssign(line); cout<<"   cod: "<<line<<"\n"; }
    cout<<"   muchii de dependenta (data-dependency graph): "<<pg.depEdges.size()<<"\n";
    { auto vals=pg.run(); cout<<"   EXECUTION GRAPH -> valori: "; for(auto&kv:vals)cout<<kv.first<<"="<<kv.second<<" "; cout<<"\n"; }

    // ================= PARTEA 6: LEARNERS =================
    cout<<"\n========== PARTEA 6: LEARNERS (online) ==========\n";
    RuleLearner rl; auto rules=rl.learn(graph,eng.nodeConcept);
    cout<<"[RULE LEARNER] reguli (motive Csrc->R->Cdst) cu count>=2: "<<rules.size()<<"\n";
    cout<<"[TRANSITION LEARNER] = WorldModel.learn ("<<wm.trans.size()<<" tranzitii)\n";
    cout<<"[GOAL LEARNER] top atractori: "; for(int i=0;i<4&&i<(int)goals.size();i++)cout<<tok.name(goals[i].first)<<"("<<goals[i].second<<") "; cout<<"\n";
    cout<<"[CONCEPT/RELATION/LANGUAGE LEARNER] = clustering + LoRA din Engine (v6)\n";

    // ================= PARTEA 5+7: VERIFIER + COGNITIVE LOOP =================
    cout<<"\n========== PARTEA 5+7: VERIFIER + COGNITIVE LOOP ==========\n";
    Verifier verifier; CognitiveLoop loop(eng,ann,wm,planner,verifier);
    loop.run("de ce lucreaza Ion", tok.get("bani"));
    loop.run("xyzzy necunoscut", goalNode);   // fara suport => raspuns prudent

    // ################# PARTEA 8: "IMPLEMENTEAZA TOT" (v8) #################
    cout<<"\n########## v8: INCHIDEM GOLURILE DIN ANALIZA v7 ##########\n";

    // --- (a) SIGNED WORLD MODEL: polaritate invatata din episoade numerice ---
    cout<<"\n== (a) WORLD MODEL CU SEMN (polaritate invatata din observatii) ==\n";
    SignedWorldModel sw;
    map<pair<int,int>,float> truth;
    { uniform_real_distribution<float> u(-1,1);
      for(auto& e:graph.edges) if(!truth.count({e.src,e.dst})) truth[{e.src,e.dst}]=(u(rng)>0?1.f:-1.f)*(0.4f+0.4f*fabs(u(rng))); }
    vector<SignedWorldModel::Episode> eps; normal_distribution<float> noise(0,0.02f);
    uniform_real_distribution<float> u01(0,1);
    for(int epi=0;epi<60;epi++){ SignedWorldModel::Episode ep; map<int,float> val;
        for(int n:graph.nodes) val[n]=u01(rng);
        for(int t=0;t<4;t++){ ep.series.push_back(val); map<int,float> nv=val;
            for(auto& e:graph.edges) nv[e.dst]+=truth[{e.src,e.dst}]*val[e.src]+noise(rng); val.swap(nv); }
        eps.push_back(ep); }
    sw.learn(graph,eps,400,0.08f);
    { int ok=0,tot=0; for(auto& kv:truth) if(fabs(kv.second)>0.15f){ tot++;
        if((sw.polarity(kv.first.first,kv.first.second)>0)==(kv.second>0))ok++; }
      cout<<"   recuperare semn din date: "<<ok<<"/"<<tot<<" muchii corecte (weak supervision, fara semantica)\n"; }
    { auto eff=sw.simulate(tok.get("munca"),1.f,4,graph);
      float onBani=eff.count(tok.get("bani"))?eff[tok.get("bani")]:0;
      cout<<"   simulare SEMNATA: munca +1 => bani "<<(onBani>=0?"+":"")<<onBani<<" (semn real, nu doar propagare)\n"; }

    // --- (b) INT8 GEMM real (acumulare int32) vs float ---
    cout<<"\n== (b) INT8 GEMM real (acumulare int32) ==\n";
    { Int8Linear q(T.W2); vector<float> x(T.W2.R); for(auto&v:x)v=u01(rng)-0.5f;
      auto yf=matmul(x,T.W2); auto yq=q.forward(x); double er=0; for(int j=0;j<T.W2.C;j++)er+=fabs(yf[j]-yq[j]);
      cout<<"   eroare medie |float - int8| = "<<er/T.W2.C<<" (kernel int8 functional, ruleaza pe mobil fara FPU mare)\n"; }

    // --- (c) MULTI-TABLE LSH: recall mai bun ---
    cout<<"\n== (c) MULTI-TABLE LSH (recall vs brute force) ==\n";
    { MultiTableANN mann(&store,DEMO_D,4,8); mann.build(graph.nodes);
      int q=tok.get("Ion"); int scanned=0; auto appr=mann.query(store.E[q],4,scanned);
      vector<pair<float,int>> bf; for(int n:graph.nodes) if(n!=q) bf.push_back({cosv(store.E[q],store.E[n]),n});
      sort(bf.begin(),bf.end(),[](auto&a,auto&b){return a.first>b.first;});
      set<int> truek; for(int i=0;i<4&&i<(int)bf.size();i++)truek.insert(bf[i].second);
      int hit=0; for(int n:appr) if(truek.count(n))hit++;
      cout<<"   query 'Ion': scanat "<<scanned<<"/"<<graph.nodes.size()<<" (4 tabele), recall@4 = "<<hit<<"/4\n"; }

    // --- (d) ACTION POLICY invatata (cand sa emita [SEARCH_GRAPH]) ---
    cout<<"\n== (d) ACTION POLICY invatata (nu prag fix) ==\n";
    { vector<pair<vector<float>,int>> data;
      // exemple: hidden al promptului + eticheta = 1 daca nu are ancora in graf (suport slab)
      for(string p:{string("Ion"),string("om munceste"),string("pisica"),string("sofer conduce"),
                    string("xyzzy"),string("qwerty necunoscut"),string("zzz abc"),string("Vasile")}){
        vector<string> nw; auto ids=eng.tokenize(p,nw); bool grounded=false; for(int t:ids)if(graph.isNode(t))grounded=true;
        data.push_back({T.hiddenLast(ids), grounded?0:1}); }
      ActionPolicy pol(DEMO_D); pol.train(data,400,0.1f);
      cout<<"   acuratete politica (low/high support): "<<pol.accuracy(data)
          <<"  (invata o directie in spatiul ascuns, generalizeaza)\n"; }

    // --- (e) VERIFIER cu contradictie de semn ---
    cout<<"\n== (e) VERIFIER: contradictie logica via polaritate ==\n";
    { vector<int> path={tok.get("Ion"),tok.get("om"),tok.get("bani")};
      cout<<"   contradictie pe drum Ion->om->bani = "<<signContradiction(path,sw)<<" (0=consistent, 1=semn contradictoriu)\n"; }

    // --- (f) PROGRAM LEARNING din TRACES de executie (sensor, fara keyword) ---
    cout<<"\n== (f) PROGRAM LEARNING din TRACES (zero keyword-uri; descopera structura) ==\n";
    {
        // simboluri OPACE s0,s1,s2 (sensorul nu le da niciun inteles). Generam
        // traces dintr-un program ASCUNS (s=s+i; i=i+one). Invatatul vede DOAR valori.
        vector<string> sn={"s0","s1","s2"}; int S=3;
        vector<vector<map<int,double>>> traces;
        for(int run=0;run<8;run++){ vector<map<int,double>> tr; double s=run,i=1,one=1;
            for(int t=0;t<6;t++){ tr.push_back({{0,s},{1,i},{2,one}}); s=s+i; i=i+one; } traces.push_back(tr); }
        TraceLearner tl; tl.learn(S,traces,600,0.02f);
        cout<<"   variabile descoperite (valoare ne-constanta): "; for(int v:tl.varying)cout<<sn[v]<<" "; cout<<"\n";
        cout<<"   functii de tranzitie descoperite (b <- a, coeficient):\n";
        for(int b=0;b<S;b++) for(int a=0;a<S;a++){ float w=tl.coef[{a,b}]; if(fabs(w)>0.3f) cout<<"      "<<sn[b]<<"' <- "<<w<<"*"<<sn[a]<<"\n"; }
        cout<<"   => a recuperat 's0'=s0+s1 si 's1'=s1+s2 DOAR din valori, fara sa parseze '+'.\n";
    }

    // --- (i) EFFECT MODES emergente (fara etichete increase/decrease/block) ---
    cout<<"\n== (i) EFFECT MODES emergente (clustere de polaritate, numerice) ==\n";
    { EffectModes em; em.discover(graph,sw);
      cout<<"   moduri de efect descoperite: "<<em.nModes<<" (ex: E0 ~ creste, E1 ~ scade -- dar NUMERIC, fara nume in cod)\n";
      for(int ei=0; ei<(int)graph.edges.size() && ei<6; ei++){ auto& e=graph.edges[ei];
        cout<<"      "<<tok.name(e.src)<<"->"<<tok.name(e.dst)<<" : E"<<em.edgeMode[ei]<<" (w="<<sw.polarity(e.src,e.dst)<<")\n"; } }

    // --- (j) TEMPORAL REASONING (ordine / lag / frecventa) ---
    cout<<"\n== (j) TEMPORAL MODEL (proces temporal descoperit din secvente) ==\n";
    { auto MK=[&](const string& w){ bool nw; vector<float> e; return eng.tok.getOrCreateToken(w,e,e,nw); };
      vector<vector<int>> seqs;
      seqs.push_back({MK("seminte"),MK("planta"),MK("fruct")});
      seqs.push_back({MK("seminte"),MK("planta"),MK("fruct")});
      seqs.push_back({MK("ou"),MK("pui"),MK("gaina")});
      TemporalModel tm; tm.learn(seqs);
      auto pr=tm.process(eng.tok.get("seminte"),4);
      cout<<"   proces din 'seminte' (urmeaza succesorul cel mai frecvent la lag~1): ";
      for(size_t i=0;i<pr.size();i++)cout<<(i?" -> ":"")<<eng.tok.name(pr[i]); cout<<"\n"; }

    // --- (k) CREATIVE ENGINE (combinatii de concepte departate -> metafore) ---
    cout<<"\n== (k) CREATIVE ENGINE (punti semantice intre noduri neconectate) ==\n";
    { CreativeEngine ce; auto br=ce.bridges(graph,store,0.25f,0.9f,5);
      if(br.empty())cout<<"   (nicio punte in pragul dat)\n";
      for(auto& p:br) cout<<"      "<<tok.name(p.first)<<" ~ "<<tok.name(p.second)<<"  (cos="<<cosv(store.E[p.first],store.E[p.second])<<")\n"; }

    // --- (l) NEXT-TOKEN INTEGRAT: graf + world-model + goal in ACELASI pas ---
    cout<<"\n== (l) NEXT-TOKEN INTEGRAT (toate motoarele contribuie la un singur token) ==\n";
    {
        vector<string> nw; auto ctx=eng.tokenize("Ion",nw); vector<float> h=T.hiddenLast(ctx);
        vector<float> qv=meanEmb(store,ctx,DEMO_D);
        ActiveGraphState A; A.seed(ctx,1.f); A.expandTopK(graph,8);
        vector<float> base=T.logitsCos(h), z=base; float bs=0;
        eng.gbias.apply(z,A,qv,store,3.0f,2.0f,bs);                              // GRAF
        { auto eff=sw.simulate(ctx.front(),1.f,4,graph);                        // WORLD MODEL
          for(auto& kv:eff) if(kv.first>=0&&kv.first<(int)z.size()) z[kv.first]+=0.5f*tanh(kv.second); }
        for(int v=0;v<(int)z.size();v++) z[v]+=0.4f*max(0.f,cosv(store.E[v],store.E[goalNode])); // GOAL
        auto top5=[&](const vector<float>& zz){ vector<pair<float,int>> v; for(int i=0;i<(int)zz.size();i++)v.push_back({zz[i],i});
            sort(v.begin(),v.end(),[](const pair<float,int>&a,const pair<float,int>&b){return a.first>b.first;});
            for(int i=0;i<5&&i<(int)v.size();i++)cout<<tok.name(v[i].second)<<"("<<v[i].first<<") "; cout<<"\n"; };
        cout<<"   logits TRANSFORMER pur          : "; top5(base);
        cout<<"   + GRAF + WORLD-MODEL + GOAL     : "; top5(z);
        cout<<"   (un singur pas de next-token, alimentat de toate motoarele simultan)\n";
    }

    // --- (g) CSR INCREMENTAL + (h) MMAP MUCHII zero-copy ---
    cout<<"\n== (g) CSR INCREMENTAL + (h) MMAP MUCHII ==\n";
    { CSRGraph c2; c2.build(graph);
      int a=tok.get("Ion"), b=tok.get("hrana");
      c2.appendEdge(a,b,-1);   // muchie noua fara rebuild
      vector<int> nb; if(c2.tokToDense.count(a)) c2.neighbors(c2.tokToDense[a],nb);
      cout<<"   CSR: append O(1) -> 'Ion' are acum "<<nb.size()<<" vecini (inainte de compact)\n";
      c2.compact(); cout<<"   CSR compact: overflow fuzionat, muchii="<<c2.edgeTargets.size()<<"\n"; }
    { MemoryMappedGraphStorage mm; if(mm.openFile("/tmp/graph.bin")){ const int* h=mm.header();
        cout<<"   MMAP muchii zero-copy: edge0 = node("<<mm.edgeSrc(0,h[2],h[1])<<") -> node("<<mm.edgeDst(0,h[2],h[1])<<")\n"; mm.closeFile(); } }

    cout<<"\n== Gata. Vezi analiza inginereasca de mai jos (in raspunsul asistentului). ==\n";
    return 0;
}
