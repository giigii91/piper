// ============================================================================
//  GRAPH-AUGMENTED TRANSFORMER (single file, C++17, fara librarii externe)
// ----------------------------------------------------------------------------
//  Model MIC + memorie externa in GRAF + invatare ONLINE. NU memoreaza lumea
//  in greutati; o tine in graf. La FIECARE next-token, transformerul consulta
//  graful inainte sa aleaga tokenul:
//
//    tokens -> Transformer(hidden) -> GraphAttention -> ConceptContext
//           -> FusionGate -> LogitBias -> NextToken -> update ActiveGraphState
//
//  Idei cheie (toate numerice, fara semantica hardcodata, fara dictionar):
//   * Embedding-uri TIED + STORE crescator: un cuvant nou = un rand nou de
//     embedding care intra automat si in logits (head legat). Corpul
//     transformerului (atentie/ffn) ramane INGHETAT => "fara retraining".
//   * Tokenizer ADAPTIV: cuvant nou -> token nou + nod nou; embedding initial
//     din CARACTERE (morfologie) + context + vecini din graf. Niciun <unk>.
//   * Online learning pe 3 niveluri:
//       A. lent  = greutatile transformerului (gramatica/structura) [pretrain]
//       B. rapid = graful (noduri/muchii/relatii/concepte/scopuri)
//       C. adaptiv = embedding-uri tokeni noi (corp inghetat)
//   * Concepte = clustere numerice de noduri (C0,C1..). Relatii = clustere
//     numerice de muchii (R0,R1..). Scopuri = atractori (convergenta in graf).
//   * Limba noua: cuvinte engleze devin tokeni/noduri noi; prin contexte
//     similare embeddingurile lor se aliniaza la conceptele existente — fara
//     dictionar ro-en.
//   * Anti-halucinatie: un raspuns factual primeste bias doar daca e SUSTINUT
//     de graf (activare*confidence peste prag). Altfel verdict numeric -> "Nu stiu".
//
//  Compilare:  g++ -O2 -std=c++17 concept_engine.cpp -o ce
//  Rulare:     ./ce
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <set>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>

using namespace std;
static mt19937 rng(7);

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

// ============================================================================
//  EMBEDDING STORE crescator, cu head TIED (logit = hidden . emb[token]).
//  Adaugarea unui token nou = un rand nou; intra automat in input si in logits.
//  Adam per-rand pentru update online doar pe randurile atinse.
// ============================================================================
struct EmbeddingStore {
    int D;
    vector<vector<float>> E, gM, gV, grad;
    EmbeddingStore(int d=0):D(d){}
    int add(const vector<float>& init){
        E.push_back(init); gM.push_back(vector<float>(D,0)); gV.push_back(vector<float>(D,0));
        grad.push_back(vector<float>(D,0)); return (int)E.size()-1;
    }
    int size()const{ return (int)E.size(); }
    void zerograd(){ for(auto& r:grad) fill(r.begin(),r.end(),0); }
    void stepRows(float lr,float t,float wd,const set<int>& rows){
        const float b1=0.9f,b2=0.999f,eps=1e-8f; float bc1=1-pow(b1,t),bc2=1-pow(b2,t);
        for(int i:rows){ for(int k=0;k<D;k++){
            float g=grad[i][k]; gM[i][k]=b1*gM[i][k]+(1-b1)*g; gV[i][k]=b2*gV[i][k]+(1-b2)*g*g;
            E[i][k]-=lr*(gM[i][k]/bc1/(sqrt(gV[i][k]/bc2)+eps)+wd*E[i][k]); } }
    }
};

// ============================================================================
//  Matrice cu Adam (pentru corpul transformerului)
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
//  TRANSFORMER (1 bloc) cu embedding-uri TIED din EmbeddingStore.
//  Corpul (Wq..W2, norme) e "invatarea lenta". Embeddingurile sunt in store
//  si pot fi actualizate separat (invatare adaptiva), cu corpul inghetat.
// ============================================================================
struct Transformer {
    int D,H,nH,hd; EmbeddingStore* es;
    Mat Wq,Wk,Wv,Wo,W1,Wg,W2; RMSNorm n1,n2,nf;
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
        vector<vector<float>> emb,a_n,res1,f_n,up,gate,act,ffn,out,hid;
        vector<float> a_inv,f_inv,nf_inv; vector<vector<float>> a_xn,f_xn,nf_xn;
        vector<vector<float>> q,k,vv,attn; vector<vector<float>> probs; };

    Cache forward(const vector<int>&ids){
        Cache c; c.ids=ids; int n=ids.size(); c.n=n;
        c.emb.assign(n,vector<float>(D));
        for(int t=0;t<n;t++) c.emb[t]=es->E[ids[t]];
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
        c.f_n.resize(n);c.f_inv.resize(n);c.f_xn.resize(n);c.up.resize(n);c.gate.resize(n);c.act.resize(n);c.ffn.resize(n);
        c.out.assign(n,vector<float>(D));
        for(int t=0;t<n;t++){ float inv;vector<float> xn;c.f_n[t]=n2.fwd(c.res1[t],inv,xn);c.f_inv[t]=inv;c.f_xn[t]=xn;
            c.up[t]=matmul(c.f_n[t],W1);c.gate[t]=matmul(c.f_n[t],Wg);
            c.act[t].resize(H);for(int i=0;i<H;i++)c.act[t][i]=siluf(c.up[t][i])*c.gate[t][i];
            c.ffn[t]=matmul(c.act[t],W2); for(int i=0;i<D;i++)c.out[t][i]=c.res1[t][i]+c.ffn[t][i]; }
        c.hid.resize(n);c.nf_inv.resize(n);c.nf_xn.resize(n);
        for(int t=0;t<n;t++){float inv;vector<float> xn;c.hid[t]=nf.fwd(c.out[t],inv,xn);c.nf_inv[t]=inv;c.nf_xn[t]=xn;}
        return c;
    }
    // logits TIED peste TOT vocabularul curent
    vector<float> logits(const vector<float>&h){
        int V=es->size(); vector<float> z(V,0);
        for(int v=0;v<V;v++) z[v]=dotv(h,es->E[v]); return z;
    }
    vector<float> hiddenLast(const vector<int>&ids){ if(ids.empty())return vector<float>(D,0); return forward(ids).hid.back(); }

    // un pas de antrenare next-token. updateBody=false => doar embeddingurile
    // se misca (invatare adaptiva, corp inghetat). Returneaza loss.
    float trainSeq(const vector<int>&ids, bool updateBody, float lr, float wd, float& step){
        int n=ids.size(); if(n<2) return 0; Cache c=forward(ids); int V=es->size();
        set<int> touched;
        vector<vector<float>> dHid(n,vector<float>(D,0)); float loss=0;int cnt=0;
        for(int t=0;t<n-1;t++){
            auto z=logits(c.hid[t]); auto p=softmax(z); int tg=ids[t+1];
            loss+=-log(max(p[tg],1e-9f)); cnt++;
            vector<float> dz=p; dz[tg]-=1;
            // grad spre hidden (head tied) + grad spre randuri de embedding
            for(int v=0;v<V;v++){ float d=dz[v]; if(d==0)continue;
                for(int i=0;i<D;i++){ dHid[t][i]+=d*es->E[v][i]; es->grad[v][i]+=d*c.hid[t][i]; }
                touched.insert(v); }
        }
        vector<vector<float>> dOut(n,vector<float>(D,0));
        for(int t=0;t<n-1;t++) dOut[t]=nf.bwd(dHid[t],c.nf_xn[t],c.nf_inv[t]);
        vector<vector<float>> dRes1(n,vector<float>(D,0));
        for(int t=0;t<n;t++){ for(int i=0;i<D;i++)dRes1[t][i]+=dOut[t][i];
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
        step+=1;
        es->stepRows(lr,step,wd,touched);
        if(updateBody) stepBody(lr,step,1e-3f);
        zerogradBody(); es->zerograd();
        return cnt?loss/cnt:0;
    }
};

// ============================================================================
//  TOKENIZER ADAPTIV — nu pierde cuvinte in <unk>. Cuvant nou => token nou,
//  cu embedding initial din CARACTERE (morfologie). Returneaza daca e nou.
// ============================================================================
struct AdaptiveTokenizer {
    unordered_map<string,int> id; vector<string> word; EmbeddingStore* es; int D;
    vector<vector<float>> charBasis;  // baza fixa pe octet -> vector D
    AdaptiveTokenizer(EmbeddingStore* store,int d):es(store),D(d){
        charBasis.assign(256,vector<float>(D));
        normal_distribution<float> g(0,1);
        for(int c=0;c<256;c++){ for(int k=0;k<D;k++)charBasis[c][k]=g(rng); }
    }
    // embedding morfologic: suma vectorilor de caracter + un termen pe bigrame
    vector<float> charEmbed(const string& w){
        vector<float> v(D,0);
        for(unsigned char c:w) for(int k=0;k<D;k++) v[k]+=charBasis[c][k];
        for(size_t i=0;i+1<w.size();i++){ unsigned char a=w[i],b=w[i+1]; int hsh=(a*131+b)&255;
            for(int k=0;k<D;k++) v[k]+=0.5f*charBasis[hsh][k]; }
        float s=nrm(v); for(float&x:v)x/=s; for(float&x:v)x*=0.1f; // scara mica
        return v;
    }
    int get(const string& w)const{ auto it=id.find(w); return it==id.end()?-1:it->second; }
    // intoarce id; daca e nou il creeaza (token + embedding initial din caractere)
    int obtain(const string& w, bool& isNew){
        auto it=id.find(w); if(it!=id.end()){ isNew=false; return it->second; }
        isNew=true; int x=es->add(charEmbed(w)); id[w]=x;
        if((int)word.size()<=x) word.resize(x+1); word[x]=w; return x;
    }
    string name(int i)const{ return (i>=0&&i<(int)word.size())?word[i]:"?"; }
};

// ============================================================================
//  CLUSTERER greedy (concept/relatie) — id-uri numerice emergente
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
};

// ============================================================================
//  GRAPH MEMORY (noduri=tokeni, muchii cu embedding + confidence + relCluster)
// ============================================================================
struct Edge { int src,dst,relCluster; float confidence; vector<float> emb; };
struct GraphMemory {
    vector<Edge> edges; unordered_map<int,vector<int>> outAdj, inAdj; set<int> nodes;
    int addEdge(int s,int d,int rel,float conf,const vector<float>& emb){
        for(int ei:outAdj[s]) if(edges[ei].dst==d&&edges[ei].relCluster==rel){
            edges[ei].confidence=min(1.f,edges[ei].confidence+0.03f); return ei; }
        int idx=edges.size(); edges.push_back({s,d,rel,conf,emb});
        outAdj[s].push_back(idx); inAdj[d].push_back(idx); nodes.insert(s); nodes.insert(d); return idx;
    }
    vector<int> out(int s)const{ auto it=outAdj.find(s); return it==outAdj.end()?vector<int>{}:it->second; }
    bool isNode(int id)const{ return nodes.count(id)>0; }
};

// ============================================================================
//  ACTIVE GRAPH STATE — noduri active in timpul generarii; update incremental
//  cu decay + spreading + reinforcement (fara recalcul global).
// ============================================================================
struct ActiveGraphState {
    map<int,float> act;
    void seed(const vector<int>& ns,float v=1.f){ for(int n:ns) act[n]=max(act[n],v); }
    void decay(float d=0.85f){ for(auto& kv:act) kv.second*=d; }
    void spread(const GraphMemory& g,float rate=0.4f){
        map<int,float> add;
        for(auto& kv:act){ if(kv.second<0.05f)continue;
            for(int ei:g.out(kv.first)){ const Edge& e=g.edges[ei];
                add[e.dst]+=rate*kv.second*e.confidence; } }
        for(auto& kv:add) act[kv.first]+=kv.second;
        // normalizare lina
        float mx=0; for(auto&kv:act)mx=max(mx,kv.second); if(mx>1) for(auto&kv:act)kv.second/=mx;
    }
    void reinforce(int node,float v=0.5f){ act[node]+=v; }
    vector<pair<int,float>> topk(int k)const{
        vector<pair<int,float>> v(act.begin(),act.end());
        sort(v.begin(),v.end(),[](auto&a,auto&b){return a.second>b.second;});
        if((int)v.size()>k)v.resize(k); return v;
    }
};

// ----------------------------------------------------------------------------
//  helper afisare
// ----------------------------------------------------------------------------
struct Engine; // fwd

int main(){
    const int D=32, FFN=64, HEADS=4;
    EmbeddingStore store(D);
    AdaptiveTokenizer tok(&store,D);
    Transformer T(&store,D,FFN,HEADS);
    GraphMemory graph;
    Clusterer relClust(0.55f), conClust(0.42f);
    map<int,int> nodeConcept;

    // ---- helpers de invatare ----
    auto meanVec=[&](const vector<int>& ids,int a,int b){ vector<float> v(D,0); int c=0;
        for(int i=a;i<b&&i<(int)ids.size();i++){ if(i<0)continue; for(int k=0;k<D;k++)v[k]+=store.E[ids[i]][k]; c++; }
        if(c)for(float&x:v)x/=c; return v; };

    // tokenizare adaptiva a unei propozitii; raporteaza tokenii noi
    auto tokenize=[&](const string& s, vector<string>& newWords){
        vector<int> ids; for(auto& w:splitw(s)){ bool isNew; int t=tok.obtain(w,isNew);
            if(isNew) newWords.push_back(w); ids.push_back(t);} return ids; };

    // ABSORBI o propozitie in graf (invatare RAPIDA: noduri/muchii/relatii/concepte)
    auto absorbSentence=[&](const vector<int>& ids){
        if((int)ids.size()<2) return;
        int s=ids.front(), d=ids.back();
        vector<float> remb = ids.size()>=3? meanVec(ids,1,ids.size()-1) : meanVec(ids,0,ids.size());
        int rc=relClust.assign(remb);
        vector<float> eemb=meanVec(ids,0,ids.size());
        graph.addEdge(s,d,rc,0.9f,eemb);
        for(int n:{s,d}) nodeConcept[n]=conClust.assign(store.E[n]); // concept (online)
    };

    // ---- A. INVATARE LENTA: pretrain corp + embeddings pe corpus romanesc ----
    vector<string> ro = {
        "om este fiinta", "om este viu", "pisica este animal", "caine este animal",
        "pisica mananca hrana", "caine mananca hrana", "om mananca hrana",
        "Ion este om", "Vasile este om", "Ion lucreaza sofer",
        "sofer conduce camion", "munca produce bani", "motorul produce putere",
        "focul produce caldura", "bani cumpara hrana", "hrana sustine om",
        "om munceste pentru bani"
    };
    vector<vector<int>> roIds;
    { vector<string> nw; for(auto& s:ro) roIds.push_back(tokenize(s,nw)); }

    cout<<"== A. INVATARE LENTA (pretrain corp transformer + embeddings) ==\n";
    cout<<"   vocabular initial: "<<store.size()<<" tokeni\n";
    float step=0;
    for(int ep=0;ep<400;ep++){ float tot=0;int c=0;
        vector<int> ord(roIds.size()); for(int i=0;i<(int)ord.size();i++)ord[i]=i;
        shuffle(ord.begin(),ord.end(),rng);
        for(int i:ord){ tot+=T.trainSeq(roIds[i], true, 0.01f, 1e-3f, step); c++; }
        if(ep%100==0) cout<<"   epoch "<<ep<<"  loss "<<tot/max(1,c)<<"\n";
    }
    // construieste graful din corpus
    for(auto& ids:roIds) absorbSentence(ids);
    cout<<"   graf: "<<graph.nodes.size()<<" noduri, "<<graph.edges.size()<<" muchii\n\n";

    auto showConcepts=[&](){
        map<int,vector<int>> byc; for(auto&kv:nodeConcept)byc[kv.second].push_back(kv.first);
        for(auto&kv:byc){ cout<<"     C"<<kv.first<<": "; bool f=true;
            for(int n:kv.second){cout<<(f?"":", ")<<tok.name(n);f=false;} cout<<"\n"; } };
    cout<<"== Concepte emergente dupa corpusul romanesc ==\n"; showConcepts(); cout<<"\n";

    auto neighbors=[&](const string& w,int k){
        int id=tok.get(w); if(id<0){cout<<"     (necunoscut)\n";return;}
        vector<pair<float,int>> v;
        for(int n:graph.nodes){ if(n==id)continue; v.push_back({cosv(store.E[id],store.E[n]),n}); }
        sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first>b.first;});
        cout<<"     "<<w<<" ~ "; for(int i=0;i<k&&i<(int)v.size();i++)cout<<tok.name(v[i].second)<<"("<<v[i].first<<") ";
        cout<<"\n"; };

    // ---- B+C. ONLINE: cuvinte noi inventate (corp INGHETAT) ----
    cout<<"== B. ONLINE: cuvinte noi inventate (fara retraining al corpului) ==\n";
    vector<string> invent = { "blorf este animal", "blorf mananca hrana", "blorf este viu" };
    for(auto& s:invent){
        vector<string> nw; auto ids=tokenize(s,nw);
        for(auto& w:nw) cout<<"   [TOKEN NOU] \""<<w<<"\" -> id "<<tok.get(w)<<", embedding init din caractere\n";
        // invatare adaptiva: doar embeddingurile se misca (corp inghetat).
        // wd mai mare tine norma marginita => tokenii noi nu domina logitii.
        for(int it=0;it<25;it++) T.trainSeq(ids, false, 0.012f, 6e-3f, step);
        absorbSentence(ids);
        cout<<"   [ABSORBIT] \""<<s<<"\"  -> noduri/muchii/concepte actualizate\n";
    }
    cout<<"   vecini invatati pentru cuvant nou:\n"; neighbors("blorf",4);
    cout<<"   concepte dupa cuvinte noi:\n"; showConcepts(); cout<<"\n";

    // ---- D. ONLINE: limba noua (engleza), fara dictionar ----
    cout<<"== C. ONLINE: limba noua (engleza) conectata prin context ==\n";
    vector<string> en = { "man is human", "man is living", "cat is animal",
                          "cat eats food", "driver drives truck", "work produces money" };
    for(auto& s:en){ vector<string> nw; auto ids=tokenize(s,nw);
        for(auto& w:nw) cout<<"   [TOKEN NOU en] \""<<w<<"\" id "<<tok.get(w)<<"\n";
        for(int it=0;it<25;it++) T.trainSeq(ids, false, 0.012f, 6e-3f, step);
        absorbSentence(ids);
    }
    cout<<"   aliniere cross-lingva emergenta (cosine, fara dictionar):\n";
    for(string w:{string("man"),string("cat"),string("driver"),string("work"),string("food")}) neighbors(w,3);
    cout<<"\n";

    // ============================================================
    //  GRAPH-AUGMENTED NEXT TOKEN
    //  hidden -> GraphAttention(top-k noduri) -> ConceptContext ->
    //  FusionGate (scalar numeric) -> fused -> logits TIED -> LogitBias ->
    //  argmax -> update ActiveGraphState
    // ============================================================
    auto graphAttention=[&](const vector<float>& h, const ActiveGraphState& A, int K,
                            vector<int>& topNodes, vector<float>& topW)->vector<float>{
        vector<pair<float,int>> sc;
        for(int n:graph.nodes){ float a=0; auto it=A.act.find(n); if(it!=A.act.end())a=it->second;
            float s=cosv(h,store.E[n])*(0.5f+a);   // similaritate modulata de activare
            sc.push_back({s,n}); }
        sort(sc.begin(),sc.end(),[](auto&a,auto&b){return a.first>b.first;});
        vector<float> g(D,0); float Z=0; topNodes.clear(); topW.clear();
        for(int i=0;i<K&&i<(int)sc.size();i++){ float w=max(0.f,sc[i].first); topNodes.push_back(sc[i].second);
            topW.push_back(w); for(int k=0;k<D;k++)g[k]+=w*store.E[sc[i].second][k]; Z+=w; }
        if(Z>0)for(float&x:g)x/=Z; return g;
    };

    // genereaza ghidat de graf; tipareste detaliile primului pas
    auto generate=[&](const string& prompt,int steps,bool verbose){
        vector<string> nw; auto ctx=tokenize(prompt,nw);
        // seed active state cu nodurile din prompt care exista in graf
        ActiveGraphState A; { vector<int> seed; for(int t:ctx) if(graph.isNode(t))seed.push_back(t); A.seed(seed,1.f); }
        // query embedding (pentru bias bazat pe relevanta)
        vector<float> qv=meanVec(ctx,0,ctx.size());
        string outText; vector<int> gen=ctx; float supportTotal=0; int supportSteps=0;
        map<int,int> usedCount; for(int t:ctx) usedCount[t]++;
        for(int s=0;s<steps;s++){
            A.spread(graph);
            vector<float> h=T.hiddenLast(gen);
            vector<int> topNodes; vector<float> topW;
            vector<float> g=graphAttention(h,A,5,topNodes,topW);
            // FusionGate: cat de mult contează graful (numeric, din alinierea h-g si masa activarii)
            float align=cosv(h,g); float mass=0; for(float w:topW)mass+=w;
            float gate=sigm(2.0f*align + 0.3f*mass - 0.5f);
            vector<float> fused(D); for(int k=0;k<D;k++)fused[k]=(1-gate)*h[k]+gate*g[k];
            // logits pe baza de DIRECTIE (cosine), ca un token sa nu domine prin
            // norma (decuplare de inflatia embeddingurilor invatate online).
            auto cosLogits=[&](const vector<float>& x){ int V=store.size(); vector<float> z(V);
                float xn=nrm(x); for(int v=0;v<V;v++) z[v]=8.0f*dotv(x,store.E[v])/(xn*nrm(store.E[v])+1e-9f); return z; };
            vector<float> base=cosLogits(h);          // logits din transformer pur
            vector<float> z=cosLogits(fused);         // logits dupa fuziune cu graful
            // LogitBias memory-aware: tokenii-noduri ACTIVE primesc bias pozitiv
            // (activare * relevanta query); ceilalti primesc o usoara suprimare,
            // ca generarea factuala sa stea in vecinatatea sustinuta de graf.
            float beta=4.0f, gamma=9.0f, bestSupport=0;
            for(float& zv:z) zv-=gamma;
            for(auto& kv:A.act){ int node=kv.first; if(node<0||node>=(int)z.size())continue;
                float rel=max(0.f,cosv(qv,store.E[node]));
                float sup=kv.second*rel;
                z[node]+=gamma+beta*sup; bestSupport=max(bestSupport,sup); }
            supportTotal+=bestSupport; supportSteps++;
            // penalizare de repetitie (numeric): tokenii deja folositi scad
            for(auto& kv:usedCount) if(kv.first<(int)z.size()) z[kv.first]-=1.5f*kv.second;
            int tok_best=-1; float bz=-1e9; for(int v=0;v<(int)z.size();v++){ if(v==gen.back())continue;
                if(z[v]>bz){bz=z[v];tok_best=v;} }
            if(verbose && s==0){
                cout<<"   [pas 0] gate(graf vs transformer)="<<gate<<"  align="<<align<<"\n";
                cout<<"   active top: "; for(auto&p:A.topk(5))cout<<tok.name(p.first)<<"("<<p.second<<") "; cout<<"\n";
                auto top5=[&](vector<float> zz){ vector<pair<float,int>> v; for(int i=0;i<(int)zz.size();i++)v.push_back({zz[i],i});
                    sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first>b.first;});
                    for(int i=0;i<5&&i<(int)v.size();i++)cout<<tok.name(v[i].second)<<"("<<v[i].first<<") "; cout<<"\n"; };
                cout<<"   logits TRANSFORMER pur : "; top5(base);
                cout<<"   logits + GRAPH BIAS    : "; top5(z);
            }
            if(tok_best<0)break;
            gen.push_back(tok_best); usedCount[tok_best]++;
            outText+=(outText.empty()?"":" ")+tok.name(tok_best);
            A.decay(); A.reinforce(tok_best,0.6f);   // reinforcement: tokenul confirma path-ul
        }
        float support = supportSteps? supportTotal/supportSteps : 0;
        return make_pair(outText,support);
    };

    cout<<"== GRAPH-AUGMENTED GENERATION ==\n";
    cout<<"  prompt: \"Ion\"\n";
    { auto r=generate("Ion",6,true); cout<<"  => raspuns: \""<<r.first<<"\"   support="<<r.second<<"\n\n"; }
    cout<<"  prompt: \"om munceste\"\n";
    { auto r=generate("om munceste",6,false); cout<<"  => raspuns: \""<<r.first<<"\"   support="<<r.second<<"\n\n"; }

    // ============================================================
    //  ANTI-HALUCINATIE: verdict numeric, nu hardcodat semantic.
    //  Daca suportul din graf < prag => raspunsul devine "Nu stiu".
    // ============================================================
    cout<<"== ANTI-HALUCINATIE (verdict pe scor de suport) ==\n";
    const float SUPPORT_MIN=0.12f;
    auto answer=[&](const string& prompt){
        auto r=generate(prompt,6,false);
        cout<<"  Q: \""<<prompt<<"\"  support="<<r.second;
        if(r.second<SUPPORT_MIN) cout<<"  =>  Nu stiu.\n";
        else cout<<"  =>  \""<<r.first<<"\"\n";
    };
    answer("Ion");                 // sustinut de graf
    answer("blorf");               // sustinut (invatat online)
    answer("xyzzy zextro");        // necunoscut total => Nu stiu

    return 0;
}
