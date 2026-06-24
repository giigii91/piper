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
//  MAIN — demo complet cu debug
// ============================================================================
int main(){
    Engine eng(DEMO_D);
    auto& tok=eng.tok; auto& store=eng.store; auto& graph=eng.graph; auto& T=eng.T;

    // ---- A. INVATARE LENTA: pretrain corp + embeddings pe corpus romanesc ----
    vector<string> ro = {
        "om este fiinta","om este viu","pisica este animal","caine este animal",
        "pisica mananca hrana","caine mananca hrana","om mananca hrana",
        "Ion este om","Vasile este om","Ion hraneste pisica","Ion lucreaza sofer",
        "sofer conduce camion","munca produce bani","motorul produce putere",
        "focul produce caldura","bani cumpara hrana","hrana sustine om","om munceste pentru bani"
    };
    vector<vector<int>> roIds; { for(auto& s:ro){ vector<string> nw; roIds.push_back(eng.tokenize(s,nw)); } }

    cout<<"== A. PRETRAIN (invatare lenta: corp transformer + embeddings) ==\n";
    cout<<"   D="<<DEMO_D<<" (tinta scalare TARGET_D="<<TARGET_D<<"), heads="<<HEADS
        <<", straturi(demo)="<<NLAYERS_DEMO<<"\n   vocab initial (incl. tokeni virtuali): "<<store.size()<<"\n";
    float step=0;
    for(int ep=0;ep<400;ep++){ float tot=0;int c=0; vector<int> ord(roIds.size());
        for(int i=0;i<(int)ord.size();i++)ord[i]=i; shuffle(ord.begin(),ord.end(),rng);
        for(int i:ord){ tot+=T.trainSeq(roIds[i],true,false,0.01f,1e-3f,step); c++; }
        if(ep%100==0)cout<<"   epoch "<<ep<<" loss "<<tot/max(1,c)<<"\n"; }
    for(auto& ids:roIds) eng.absorb(ids);
    cout<<"   graf: "<<graph.nodes.size()<<" noduri, "<<graph.edges.size()<<" muchii\n\n";

    auto showConcepts=[&](){ map<int,vector<int>> byc; for(auto&kv:eng.nodeConcept)byc[kv.second].push_back(kv.first);
        for(auto&kv:byc){ cout<<"     C"<<kv.first<<": "; bool f=true; for(int n:kv.second){cout<<(f?"":", ")<<tok.name(n);f=false;} cout<<"\n"; } };
    cout<<"== CONCEPTE emergente (clustere de noduri) ==\n"; showConcepts();
    cout<<"== MUCHII ORDINALE (src@pos --R--> dst@pos, offset, orderConf) ==\n";
    for(int i=0;i<(int)graph.edges.size() && i<8;i++){ auto&e=graph.edges[i];
        cout<<"     "<<tok.name(e.src)<<"@"<<(int)e.srcPosition<<" --R"<<e.relCluster<<"--> "<<tok.name(e.dst)
            <<"@"<<(int)e.dstPosition<<"  offset="<<(int)e.seqOffset<<" orderConf="<<e.orderConfidence<<"\n"; }
    cout<<"\n";

    auto neighbors=[&](const string& w,int k){ int id=tok.get(w); if(id<0){cout<<"     (necunoscut)\n";return;}
        vector<pair<float,int>> v; for(int n:graph.nodes){ if(n==id)continue; v.push_back({cosv(store.E[id],store.E[n]),n}); }
        sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first>b.first;});
        cout<<"     "<<w<<" ~ "; for(int i=0;i<k&&i<(int)v.size();i++)cout<<tok.name(v[i].second)<<"("<<v[i].first<<") "; cout<<"\n"; };

    // ---- B. ONLINE: cuvinte inventate (corp INGHETAT; doar embeddings) ----
    cout<<"== B. ONLINE: cuvinte noi inventate (fara retraining al corpului) ==\n";
    for(string s:{string("Ion hraneste blorf"),string("blorf este animal"),string("blorf mananca hrana")}){
        vector<string> nw; auto ids=eng.tokenize(s,nw);
        for(auto& w:nw) cout<<"   [TOKEN NOU] \""<<w<<"\" id "<<tok.get(w)<<" (context fast-weights: char+context+vecini)\n";
        for(int it=0;it<25;it++) T.trainSeq(ids,false,false,0.012f,6e-3f,step);
        // online embedding learner: muta nodul spre contextul propozitiei
        vector<float> ctx=meanEmb(store,ids,DEMO_D); for(int id:ids) eng.online.updateNodeEmbedding(id,ctx,0.05f);
        eng.absorb(ids);
    }
    cout<<"   vecini invatati pentru 'blorf':\n"; neighbors("blorf",4);

    // ---- C. ONLINE: limba noua (engleza) prin LoRA per-domeniu ----
    cout<<"\n== C. ONLINE: limba noua (engleza) via DomainDetector + LoRA ==\n";
    vector<string> en={"man is human","man is living","cat is animal","cat eats food",
                       "driver drives truck","work produces money"};
    // detecteaza domeniul propozitiilor engleze (distributie diferita => domeniu nou)
    eng.lora.clear();
    auto ensureDomain=[&](const vector<int>& ids)->int{ vector<float> se=meanEmb(store,ids,DEMO_D); bool isNew;
        int d=eng.domains.detect(se,isNew); while((int)eng.lora.size()<=d) eng.lora.push_back(LoRAAdapter(DEMO_D,LORA_R,1.0f));
        return d; };
    // pre-creeaza tokenii ca sa avem embedding pentru detectie de domeniu
    vector<vector<int>> enIds; for(auto& s:en){ vector<string> nw; auto ids=eng.tokenize(s,nw); enIds.push_back(ids);
        for(auto& w:nw) cout<<"   [TOKEN NOU en] \""<<w<<"\" id "<<tok.get(w)<<"\n"; }
    for(size_t i=0;i<enIds.size();i++){ int d=ensureDomain(enIds[i]); eng.T.ffnLora=&eng.lora[d];
        for(int it=0;it<25;it++) T.trainSeq(enIds[i],false,true,0.012f,6e-3f,step); // doar embeddings + LoRA-ul domeniului
        eng.absorb(enIds[i]); }
    eng.T.ffnLora=nullptr;
    cout<<"   domenii detectate (clustere de distributie): "<<eng.domains.clust.centroid.size()
        <<" (=> "<<eng.lora.size()<<" adaptere LoRA)\n";
    cout<<"   aliniere cross-lingva emergenta (cosine, fara dictionar):\n";
    for(string w:{string("man"),string("cat"),string("driver"),string("work")}) neighbors(w,3);

    // ---- INT8-ready demo (doar arata structura) ----
    { QuantizedMat q=QuantizedMat::fromMat(T.W2);
      cout<<"\n== INT8-READY (demo) ==\n   W2 cuantizat int8: "<<q.R<<"x"<<q.C
          <<", eroare medie dequant ~ "; double er=0; for(int i=0;i<T.W2.R;i++)for(int j=0;j<T.W2.C;j++)er+=fabs(q.at(i,j)-T.W2.at(i,j));
      cout<<er/(T.W2.R*T.W2.C)<<" (structura; kernel INT8 ar inlocui Mat in productie)\n"; }

    // ---- D. GRAPH-AUGMENTED GENERATION cu debug pe pasi ----
    cout<<"\n== D. GRAPH-AUGMENTED GENERATION (cross-attention + bias) ==\n";
    auto generate=[&](const string& prompt,int steps,bool factual,bool verbose){
        vector<string> nw; auto ctx=eng.tokenize(prompt,nw);
        ActiveGraphState A; { vector<int> seed; for(int t:ctx) if(graph.isNode(t))seed.push_back(t); A.seed(seed,1.f); }
        vector<float> qv=meanEmb(store,ctx,DEMO_D); string out; vector<int> gen=ctx;
        set<int> used(ctx.begin(),ctx.end()); float supSum=0; int supN=0; float firstSup=-1;
        for(int s=0;s<steps;s++){ Engine::StepDbg dbg;
            int nt=eng.nextToken(gen,A,qv,used,dbg,factual); if(nt<0)break;
            supSum+=dbg.support; supN++; if(firstSup<0)firstSup=dbg.support; // grounding = suport la pasul 0
            if(verbose && s<2){ cout<<"   [pas "<<s<<"] crossScore="<<dbg.crossScore<<" fusedNorm="<<dbg.fusedNorm
                <<" support="<<dbg.support<<"\n      active: ";
                for(auto&p:A.topk(5))cout<<tok.name(p.first)<<"("<<p.second<<") "; cout<<"\n      paths: ";
                for(auto&hp:A.activePaths){ cout<<"["; for(size_t i=0;i<hp.nodes.size();i++)cout<<(i?">":"")<<tok.name(hp.nodes[i]); cout<<"] "; } cout<<"\n";
                cout<<"      logits BEFORE bias: "; for(size_t i=0;i<dbg.baseIdx.size();i++)cout<<tok.name(dbg.baseIdx[i])<<"("<<dbg.baseTop[i]<<") "; cout<<"\n";
                cout<<"      logits AFTER  bias: "; for(size_t i=0;i<dbg.biasIdx.size();i++)cout<<tok.name(dbg.biasIdx[i])<<"("<<dbg.biasTop[i]<<") "; cout<<"\n";
                cout<<"      -> token ales: "<<tok.name(nt)<<"\n"; }
            gen.push_back(nt); used.insert(nt); out+=(out.empty()?"":" ")+tok.name(nt);
            A.decay(); A.reinforceByToken(nt,store,0.6f); }
        // grounding (suport la pasul 0, inainte de auto-reinforcement) = verdictul
        // factual; media ar fi inflata de tokenii auto-generati.
        float sup=firstSup<0?0:firstSup; return make_pair(out,sup);
    };
    cout<<"  prompt: \"Ion\"\n"; { auto r=generate("Ion",6,true,true); cout<<"  => \""<<r.first<<"\"  support="<<r.second<<"\n\n"; }
    cout<<"  prompt: \"om munceste\"\n"; { auto r=generate("om munceste",6,true,false); cout<<"  => \""<<r.first<<"\"  support="<<r.second<<"\n\n"; }

    // ---- E. COGNITIVE LOOP cu tokeni virtuali ----
    //  Mecanism general: daca suportul initial e slab (incertitudine), modelul
    //  emite [SEARCH_GRAPH]; bucla cauta un path, il INJECTEAZA in context si
    //  continua. Trigger-ul e NUMERIC (prag de suport), nu semantic.
    cout<<"== E. COGNITIVE LOOP (tokeni virtuali de actiune) ==\n";
    auto cognitive=[&](const string& prompt){
        vector<string> nw; auto ctx=eng.tokenize(prompt,nw);
        ActiveGraphState A; { vector<int> seed; for(int t:ctx) if(graph.isNode(t))seed.push_back(t); A.seed(seed,1.f); }
        vector<float> qv=meanEmb(store,ctx,DEMO_D); Engine::StepDbg dbg; set<int> used(ctx.begin(),ctx.end());
        int first=eng.nextToken(ctx,A,qv,used,dbg,true);
        cout<<"  prompt: \""<<prompt<<"\"  support initial="<<dbg.support<<"\n";
        if(dbg.support < 0.15f){ // incertitudine -> actiune [SEARCH_GRAPH]
            cout<<"  model emite "<<tok.name(eng.VT_SEARCH)<<" (suport slab => cauta in graf)\n";
            A.buildHypotheses(graph,1,6);
            if(!A.activePaths.empty()){ cout<<"  [engine] injecteaza path: ";
                for(size_t i=0;i<A.activePaths[0].nodes.size();i++)cout<<(i?" > ":"")<<tok.name(A.activePaths[0].nodes[i]); cout<<"\n";
                for(int n:A.activePaths[0].nodes) ctx.push_back(n); }
            cout<<"  model emite "<<tok.name(eng.VT_RESULT)<<" si continua generarea\n";
        } else cout<<"  suport suficient => fara cautare; genereaza direct ("<<tok.name(first)<<"...)\n";
    };
    cognitive("Ion");
    cognitive("xyzzy");   // necunoscut => suport slab => [SEARCH_GRAPH] (dar graf gol pe acel nod)

    // ---- F. ANTI-HALUCINATIE pe suport factual (verdict numeric) ----
    cout<<"\n== F. ANTI-HALUCINATIE (verdict pe suport, nu pe cuvinte) ==\n";
    const float SUP_MIN=0.12f;
    auto answer=[&](const string& p){ auto r=generate(p,5,true,false);
        cout<<"  Q:\""<<p<<"\" support="<<r.second<<(r.second<SUP_MIN?"  => Nu stiu (prudent).\n":("  => \""+r.first+"\"\n")); };
    answer("Ion"); answer("blorf"); answer("xyzzy zextro");
    return 0;
}
