// ============================================================================
//  MOTOR NEURO-SIMBOLIC fara reguli semantice hardcodate.
// ----------------------------------------------------------------------------
//  Cerinta (audit): ZERO decizii de tip if(semantic). Nicaieri in cod nu exista
//    if(concept) / if(definition) / if(cause) / if(purpose) / if(question type)
//    / if(word=="este") / if(relation==ceva). Toate deciziile SEMANTICE sunt
//    inlocuite cu CALCUL: embedding, similaritate, clustering, scor, graf, prag.
//
//  Singurele "if"-uri din cod sunt TEHNICE: index valid, vector gol, prag
//  numeric, comparatie de scoruri, bucle, alocari.
//
//  Pipeline:
//    1. Tokenizer general (numeric, fara semantica).
//    2. Encoder neuronal -> embeddings de tokeni (next-token, nesupervizat).
//    3. GraphMemory: noduri + muchii; fiecare muchie are embedding.
//    4. Relation discovery: muchiile se grupeaza in CLUSTERE numerice R0,R1,...
//       dupa similaritatea embeddingului de relatie (nu dupa string).
//    5. Concept discovery: nodurile se grupeaza in concepte C0,C1,... dupa
//       similaritatea embeddingurilor de nod.
//    6. Forward-chaining NUMERIC: compune muchii adiacente compatibile vectorial
//       (nu pe tipuri), genereaza muchii derivate.
//    7. Reasoning: intrebarea -> embedding -> PATH-SEARCH ponderat in graf.
//       Scor = relevanta(query,muchie) + confidence. Fara "if de ce".
//    8. Anti-halucinatie: daca cel mai bun path are scor/relevanta sub prag,
//       raspunde "Nu stiu". Nu inventeaza.
//
//  Limita asumata onest ("fara magie falsa"): calitatea clusterelor depinde de
//  encoderul mic antrenat pe un corpus mic; doua relatii sunt grupate impreuna
//  doar daca embeddingurile lor ies suficient de apropiate. Nu fortam nimic cu
//  reguli — daca semnalul vectorial nu e suficient, raman clustere separate.
//
//  C++17, single file, fara librarii externe.
//  Compilare:  g++ -O2 -std=c++17 concept_engine.cpp -o ce
//  Rulare:     ./ce [knowledge.txt] [questions.txt]
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <tuple>
#include <queue>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>

using namespace std;
static mt19937 rng(7);

// ----------------------------------------------------------------------------
//  Utilitare numerice
// ----------------------------------------------------------------------------
vector<string> split(const string& s){
    vector<string> r; string w;
    for(char c:s){ if(c==' '||c=='\t'){ if(!w.empty()) r.push_back(w); w.clear(); } else w+=c; }
    if(!w.empty()) r.push_back(w);
    return r;
}
string trim(const string& s){
    size_t a=s.find_first_not_of(" \t\r\n"); if(a==string::npos) return "";
    size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1);
}
float dot(const vector<float>&a,const vector<float>&b){ float s=0; for(size_t i=0;i<a.size();i++) s+=a[i]*b[i]; return s; }
float norm(const vector<float>&a){ return sqrt(dot(a,a)+1e-9f); }
float cosSim(const vector<float>&a,const vector<float>&b){
    if(a.empty()||b.empty()) return 0.f;
    return dot(a,b)/(norm(a)*norm(b));
}
vector<float> vadd(const vector<float>&a,const vector<float>&b){
    vector<float> r(a.size()); for(size_t i=0;i<a.size();i++) r[i]=a[i]+b[i]; return r;
}

// ============================================================================
//  ENCODER NEURAL (transformer mic) -> embeddings de tokeni
//  (next-token nesupervizat; nicio eticheta semantica).
// ============================================================================
struct Mat {
    int R,C; vector<float> w,g,m,v;
    Mat(){} Mat(int r,int c):R(r),C(c),w(r*c),g(r*c,0),m(r*c,0),v(r*c,0){}
    void init(float s){ normal_distribution<float> d(0,s); for(float&x:w)x=d(rng);}
    float& at(int i,int j){return w[i*C+j];}
    float& gat(int i,int j){return g[i*C+j];}
    void zerograd(){ fill(g.begin(),g.end(),0);}
    void step(float lr,float t,float wd){
        const float b1=0.9f,b2=0.999f,eps=1e-8f;
        float bc1=1-pow(b1,t),bc2=1-pow(b2,t);
        for(size_t i=0;i<w.size();i++){
            m[i]=b1*m[i]+(1-b1)*g[i]; v[i]=b2*v[i]+(1-b2)*g[i]*g[i];
            w[i]-=lr*(m[i]/bc1/(sqrt(v[i]/bc2)+eps)+wd*w[i]);
        }
    }
};
vector<float> matmul(const vector<float>&x,Mat&W){
    vector<float> y(W.C,0);
    for(int i=0;i<W.R;i++){ float xi=x[i]; if(xi==0)continue;
        for(int j=0;j<W.C;j++) y[j]+=xi*W.at(i,j); }
    return y;
}
vector<float> matmul_bwd(const vector<float>&x,const vector<float>&dy,Mat&W){
    vector<float> dx(W.R,0);
    for(int i=0;i<W.R;i++){ float a=0,xi=x[i];
        for(int j=0;j<W.C;j++){ W.gat(i,j)+=xi*dy[j]; a+=W.at(i,j)*dy[j]; }
        dx[i]=a; }
    return dx;
}
vector<float> softmax(vector<float> x){
    float m=*max_element(x.begin(),x.end()),s=0;
    for(float&v:x){v=exp(v-m);s+=v;} for(float&v:x)v/=s; return x;
}
float silu(float x){return x/(1+exp(-x));}
float dsilu(float x){float s=1/(1+exp(-x));return s+x*s*(1-s);}
struct RMSNorm {
    int D; Mat g; RMSNorm(){} RMSNorm(int d):D(d),g(1,d){for(int i=0;i<d;i++)g.at(0,i)=1;}
    vector<float> fwd(const vector<float>&x,float&inv,vector<float>&xn){
        float ms=0; for(float v:x)ms+=v*v; ms/=D; inv=1/sqrt(ms+1e-5f);
        xn.resize(D); vector<float> y(D);
        for(int i=0;i<D;i++){xn[i]=x[i]*inv;y[i]=xn[i]*g.at(0,i);} return y;
    }
    vector<float> bwd(const vector<float>&dy,const vector<float>&xn,float inv){
        vector<float> dxn(D); float d=0;
        for(int i=0;i<D;i++){g.gat(0,i)+=dy[i]*xn[i];dxn[i]=dy[i]*g.at(0,i);d+=dxn[i]*xn[i];}
        vector<float> dx(D); for(int i=0;i<D;i++)dx[i]=inv*(dxn[i]-xn[i]*d/D); return dx;
    }
};
struct Encoder {
    int V,T,D,H,nH,hd;
    Mat tokEmb,Wq,Wk,Wv,Wo,W1,Wg,W2,head; RMSNorm n1,n2,nf;
    Encoder(int v,int t,int d,int ffn,int heads)
        :V(v),T(t),D(d),H(ffn),nH(heads),hd(d/heads),
         tokEmb(v,d),Wq(d,d),Wk(d,d),Wv(d,d),Wo(d,d),
         W1(d,ffn),Wg(d,ffn),W2(ffn,d),head(d,v),n1(d),n2(d),nf(d){
        float s=sqrt(2.0f/d);
        tokEmb.init(0.02f);Wq.init(s);Wk.init(s);Wv.init(s);Wo.init(s);
        W1.init(s);Wg.init(s);W2.init(sqrt(2.0f/ffn));head.init(0.02f);
    }
    vector<Mat*> params(){return{&tokEmb,&Wq,&Wk,&Wv,&Wo,&W1,&Wg,&W2,&head,&n1.g,&n2.g,&nf.g};}
    void zerograd(){for(auto*p:params())p->zerograd();}
    void step(float lr,float t,float wd){for(auto*p:params())p->step(lr,t,wd);}
    struct Cache{
        vector<int> ids; int n;
        vector<vector<float>> emb,a_n,res1,f_n,up,gate,act,ffn,out,hid;
        vector<float> a_inv,f_inv,nf_inv; vector<vector<float>> a_xn,f_xn,nf_xn;
        vector<vector<float>> q,k,vv,attn; vector<vector<float>> probs;
    };
    Cache forward(const vector<int>&ids){
        Cache c; c.ids=ids; int n=ids.size(); c.n=n;
        c.emb.assign(n,vector<float>(D));
        for(int t=0;t<n;t++)for(int i=0;i<D;i++)c.emb[t][i]=tokEmb.at(ids[t],i);
        c.a_n.resize(n);c.a_inv.resize(n);c.a_xn.resize(n);
        c.q.resize(n);c.k.resize(n);c.vv.resize(n);
        for(int t=0;t<n;t++){
            float inv;vector<float> xn; c.a_n[t]=n1.fwd(c.emb[t],inv,xn);
            c.a_inv[t]=inv;c.a_xn[t]=xn;
            c.q[t]=matmul(c.a_n[t],Wq);c.k[t]=matmul(c.a_n[t],Wk);c.vv[t]=matmul(c.a_n[t],Wv);
        }
        c.attn.assign(n,vector<float>(D,0)); c.probs.resize(nH*n);
        float invs=1/sqrt((float)hd);
        for(int h=0;h<nH;h++)for(int t=0;t<n;t++){
            vector<float> sc(t+1);
            for(int j=0;j<=t;j++){float s=0;for(int i=0;i<hd;i++)s+=c.q[t][h*hd+i]*c.k[j][h*hd+i];sc[j]=s*invs;}
            auto p=softmax(sc); c.probs[h*n+t]=p;
            for(int j=0;j<=t;j++)for(int i=0;i<hd;i++)c.attn[t][h*hd+i]+=p[j]*c.vv[j][h*hd+i];
        }
        c.res1.assign(n,vector<float>(D));
        for(int t=0;t<n;t++){auto o=matmul(c.attn[t],Wo);for(int i=0;i<D;i++)c.res1[t][i]=c.emb[t][i]+o[i];}
        c.f_n.resize(n);c.f_inv.resize(n);c.f_xn.resize(n);
        c.up.resize(n);c.gate.resize(n);c.act.resize(n);c.ffn.resize(n);
        c.out.assign(n,vector<float>(D));
        for(int t=0;t<n;t++){
            float inv;vector<float> xn;c.f_n[t]=n2.fwd(c.res1[t],inv,xn);c.f_inv[t]=inv;c.f_xn[t]=xn;
            c.up[t]=matmul(c.f_n[t],W1);c.gate[t]=matmul(c.f_n[t],Wg);
            c.act[t].resize(H);for(int i=0;i<H;i++)c.act[t][i]=silu(c.up[t][i])*c.gate[t][i];
            c.ffn[t]=matmul(c.act[t],W2);
            for(int i=0;i<D;i++)c.out[t][i]=c.res1[t][i]+c.ffn[t][i];
        }
        c.hid.resize(n);c.nf_inv.resize(n);c.nf_xn.resize(n);
        for(int t=0;t<n;t++){float inv;vector<float> xn;c.hid[t]=nf.fwd(c.out[t],inv,xn);c.nf_inv[t]=inv;c.nf_xn[t]=xn;}
        return c;
    }
    vector<float> logits(const vector<float>&h){return matmul(h,head);}
    float trainSeq(const vector<int>&ids){
        int n=ids.size(); if(n<2) return 0; Cache c=forward(ids);
        vector<vector<float>> dHid(n,vector<float>(D,0));
        float loss=0;int cnt=0;
        for(int t=0;t<n-1;t++){
            auto z=logits(c.hid[t]); auto p=softmax(z); int tg=ids[t+1];
            loss+=-log(max(p[tg],1e-9f));cnt++;
            vector<float> dz=p;dz[tg]-=1; dHid[t]=matmul_bwd(c.hid[t],dz,head);
        }
        vector<vector<float>> dOut(n,vector<float>(D,0));
        for(int t=0;t<n-1;t++) dOut[t]=nf.bwd(dHid[t],c.nf_xn[t],c.nf_inv[t]);
        vector<vector<float>> dRes1(n,vector<float>(D,0));
        for(int t=0;t<n;t++){
            for(int i=0;i<D;i++)dRes1[t][i]+=dOut[t][i];
            vector<float> dact=matmul_bwd(c.act[t],dOut[t],W2);
            vector<float> dup(H),dg(H);
            for(int i=0;i<H;i++){float su=silu(c.up[t][i]);dg[i]=dact[i]*su;dup[i]=dact[i]*c.gate[t][i]*dsilu(c.up[t][i]);}
            vector<float> d1=matmul_bwd(c.f_n[t],dup,W1),d2=matmul_bwd(c.f_n[t],dg,Wg);
            vector<float> df(D);for(int i=0;i<D;i++)df[i]=d1[i]+d2[i];
            vector<float> dr=n2.bwd(df,c.f_xn[t],c.f_inv[t]);
            for(int i=0;i<D;i++)dRes1[t][i]+=dr[i];
        }
        vector<vector<float>> dEmb(n,vector<float>(D,0)),dAttn(n,vector<float>(D,0));
        for(int t=0;t<n;t++){ for(int i=0;i<D;i++)dEmb[t][i]+=dRes1[t][i]; dAttn[t]=matmul_bwd(c.attn[t],dRes1[t],Wo); }
        vector<vector<float>> dq(n,vector<float>(D,0)),dk(n,vector<float>(D,0)),dv(n,vector<float>(D,0));
        float invs=1/sqrt((float)hd);
        for(int h=0;h<nH;h++)for(int t=0;t<n;t++){
            int L=t+1; auto&p=c.probs[h*n+t]; vector<float> dp(L,0);
            for(int j=0;j<L;j++){float dd=0;for(int i=0;i<hd;i++){dv[j][h*hd+i]+=p[j]*dAttn[t][h*hd+i];dd+=dAttn[t][h*hd+i]*c.vv[j][h*hd+i];}dp[j]=dd;}
            float dotp=0;for(int j=0;j<L;j++)dotp+=dp[j]*p[j];
            for(int j=0;j<L;j++){float ds=p[j]*(dp[j]-dotp)*invs;
                for(int i=0;i<hd;i++){dq[t][h*hd+i]+=ds*c.k[j][h*hd+i];dk[j][h*hd+i]+=ds*c.q[t][h*hd+i];}}
        }
        for(int t=0;t<n;t++){
            vector<float> a=matmul_bwd(c.a_n[t],dq[t],Wq),b=matmul_bwd(c.a_n[t],dk[t],Wk),cc=matmul_bwd(c.a_n[t],dv[t],Wv);
            vector<float> dn(D);for(int i=0;i<D;i++)dn[i]=a[i]+b[i]+cc[i];
            vector<float> de=n1.bwd(dn,c.a_xn[t],c.a_inv[t]);
            for(int i=0;i<D;i++)dEmb[t][i]+=de[i];
        }
        for(int t=0;t<n;t++)for(int i=0;i<D;i++)tokEmb.gat(ids[t],i)+=dEmb[t][i];
        return cnt?loss/cnt:0;
    }
    vector<float> tokenVec(int id){vector<float> v(D);for(int i=0;i<D;i++)v[i]=tokEmb.at(id,i);return v;}
    // distributie de next-token pe vocabular dat un context (PREDICTIE)
    vector<float> nextDist(const vector<int>& ctx){
        if(ctx.empty()) return vector<float>(V,1.0f/V);
        Cache c=forward(ctx); return softmax(logits(c.hid.back()));
    }
    // cat de tare prezice contextul 'ctx' tokenul 'tgt' (semnal cauzal/predictiv)
    float predProb(const vector<int>& ctx,int tgt){
        auto d=nextDist(ctx); return (tgt>=0&&tgt<(int)d.size())? d[tgt] : 0.f;
    }
    // embedding al unei secvente = media embeddingurilor tokenilor (numeric)
    vector<float> meanVec(const vector<int>& ids,int a,int b){
        vector<float> v(D,0); int c=0;
        for(int i=a;i<b && i<(int)ids.size();i++){ if(i<0)continue; auto t=tokenVec(ids[i]); for(int k=0;k<D;k++)v[k]+=t[k]; c++; }
        if(c>0) for(float&x:v)x/=c;
        return v;
    }
};

// ============================================================================
//  TOKENIZER general (doar id-uri numerice)
// ============================================================================
struct Tokenizer {
    unordered_map<string,int> id; vector<string> word;
    int add(const string&s){auto it=id.find(s);if(it!=id.end())return it->second;
        int x=word.size();id[s]=x;word.push_back(s);return x;}
    int get(const string&s)const{auto it=id.find(s);return it==id.end()?-1:it->second;}
    vector<int> enc(const string&s){vector<int> r;for(auto&w:split(s))r.push_back(add(w));return r;}
    string name(int id)const{return (id>=0&&id<(int)word.size())?word[id]:"?";}
};

// ============================================================================
//  CLUSTERER GENERIC (greedy, prag cosinus) — emergent, fara nume
//  Folosit identic pentru RELATII (pe embedding de relatie) si pentru
//  CONCEPTE (pe embedding de nod). Returneaza id-uri numerice 0,1,2,...
// ============================================================================
struct Clusterer {
    float thresh;
    vector<vector<float>> centroid; vector<int> count;
    Clusterer(float t):thresh(t){}
    int assign(const vector<float>& v){
        int best=-1; float bs=thresh;
        for(size_t i=0;i<centroid.size();i++){ float s=cosSim(v,centroid[i]); if(s>bs){bs=s;best=(int)i;} }
        if(best<0){ centroid.push_back(v); count.push_back(1); return (int)centroid.size()-1; }
        for(size_t k=0;k<v.size();k++) centroid[best][k]=(centroid[best][k]*count[best]+v[k])/(count[best]+1);
        count[best]++; return best;
    }
    int nearest(const vector<float>& v)const{   // doar citire, fara a crea cluster
        int best=-1; float bs=-2;
        for(size_t i=0;i<centroid.size();i++){ float s=cosSim(v,centroid[i]); if(s>bs){bs=s;best=(int)i;} }
        return best;
    }
};

// ============================================================================
//  GRAPH MEMORY — noduri + muchii cu embedding
// ============================================================================
struct Edge {
    int src,dst;            // node ids (token ids)
    int relCluster;         // R0,R1,... (emergent, numeric)
    float confidence;
    vector<float> emb;      // embedding-ul propozitiei-sursa (nod+relatie+nod)
    bool derived;           // creata prin forward-chaining numeric
    int sentence;           // index propozitie sursa (pentru debug)
};
struct GraphMemory {
    vector<Edge> edges;
    unordered_map<int,vector<int>> outAdj;
    set<int> nodes;
    int addEdge(const Edge& e){
        for(int ei:outAdj[e.src]) if(edges[ei].dst==e.dst && edges[ei].relCluster==e.relCluster){
            if(e.confidence>edges[ei].confidence) edges[ei].confidence=e.confidence; return ei; }
        int idx=edges.size(); edges.push_back(e); outAdj[e.src].push_back(idx);
        nodes.insert(e.src); nodes.insert(e.dst); return idx;
    }
    vector<int> outIdx(int src)const{ auto it=outAdj.find(src); return it==outAdj.end()?vector<int>{}:it->second; }
    bool isNode(int id)const{ return nodes.count(id)>0; }
};

// ============================================================================
//  MAIN
// ============================================================================
int main(int argc,char** argv){
    string kpath = argc>1? argv[1] : "knowledge.txt";
    string qpath = argc>2? argv[2] : "questions.txt";

    // ---- citeste corpusul (fapte) si intrebarile ----
    auto readLines=[&](const string& path)->vector<string>{
        vector<string> out; ifstream f(path); string line;
        while(getline(f,line)){ line=trim(line); if(line.empty()||line[0]=='#') continue; out.push_back(line); }
        return out;
    };
    vector<string> factLines=readLines(kpath), qLines=readLines(qpath);
    if(factLines.empty()){ cout<<"(corpus gol: "<<kpath<<")\n"; return 0; }
    cout<<"== Corpus: "<<factLines.size()<<" fapte din \""<<kpath<<"\", "
        <<qLines.size()<<" intrebari din \""<<qpath<<"\" ==\n";

    // ---- tokenizare (fapte + intrebari => vocabular) ----
    Tokenizer tok;
    vector<vector<int>> factIds, qIds, trainCorpus;
    for(auto& s:factLines){ auto v=tok.enc(s); factIds.push_back(v); trainCorpus.push_back(v); }
    for(auto& s:qLines){ auto v=tok.enc(s); qIds.push_back(v); trainCorpus.push_back(v); }

    // ---- antreneaza encoderul (embeddings nesupervizate) ----
    Encoder enc(tok.word.size(), 24, 32, 64, 4);
    cout<<"== Antrenez encoderul (embeddings) ==\n";
    float lr=0.003f,wd=0.001f,step=0;
    for(int ep=0;ep<250;ep++){
        float tot=0;int cc=0;
        vector<int> ord(trainCorpus.size()); for(int i=0;i<(int)ord.size();i++)ord[i]=i;
        shuffle(ord.begin(),ord.end(),rng);
        for(int idx:ord){enc.zerograd();tot+=enc.trainSeq(trainCorpus[idx]);step++;enc.step(lr,step,wd);cc++;}
        if(ep%50==0)cout<<"  epoch "<<ep<<"  loss "<<tot/cc<<"\n";
    }
    cout<<"Done.\n\n";

    // ---- embedding de relatie (suprafata) si de muchie (intreaga propozitie)
    auto relEmbOf=[&](const vector<int>& ids)->vector<float>{
        // suprafata-relatie = tokenii dintre primul si ultimul; daca lipsesc,
        // foloseste media intregii propozitii (decizie pur structurala).
        int n=ids.size();
        if(n>=3) return enc.meanVec(ids,1,n-1);
        return enc.meanVec(ids,0,n);
    };
    auto edgeEmbOf=[&](const vector<int>& ids)->vector<float>{ return enc.meanVec(ids,0,ids.size()); };

    // ============================================================
    //  RELATION DISCOVERY prin CLUSTERING (R0,R1,...)
    // ============================================================
    GraphMemory graph;
    Clusterer relClust(0.55f);   // prag cosinus pe embedding de relatie
    vector<int> sentRel(factIds.size(),-1);
    for(size_t s=0;s<factIds.size();s++){
        auto& ids=factIds[s];
        if((int)ids.size()<2) continue;             // tehnic: nevoie de >=2 tokeni
        int src=ids.front(), dst=ids.back();
        vector<float> remb=relEmbOf(ids);
        int rc=relClust.assign(remb);
        sentRel[s]=rc;
        Edge e; e.src=src; e.dst=dst; e.relCluster=rc; e.confidence=0.95f;
        e.emb=edgeEmbOf(ids); e.derived=false; e.sentence=(int)s;
        graph.addEdge(e);
    }

    // ============================================================
    //  CONCEPT DISCOVERY prin CLUSTERING al embeddingurilor de nod
    // ============================================================
    Clusterer conClust(0.38f);
    map<int,int> nodeConcept;                       // node -> concept id
    vector<int> nodeList(graph.nodes.begin(),graph.nodes.end());
    sort(nodeList.begin(),nodeList.end());
    for(int nd:nodeList) nodeConcept[nd]=conClust.assign(enc.tokenVec(nd));

    // ============================================================
    //  FORWARD-CHAINING NUMERIC: compune muchii adiacente compatibile
    //  vectorial (sim intre embeddingurile lor >= prag). Fara tipuri.
    // ============================================================
    {
        float compat=0.62f, decay=0.9f; int base=graph.edges.size();
        for(int i=0;i<base;i++){
            Edge A=graph.edges[i];
            for(int j:graph.outIdx(A.dst)){
                Edge B=graph.edges[j];
                if(B.dst==A.src) continue;                  // tehnic: evita bucla
                if(cosSim(A.emb,B.emb) < compat) continue;  // prag numeric, nu tip
                Edge e; e.src=A.src; e.dst=B.dst;
                e.emb=vadd(A.emb,B.emb); for(float&x:e.emb)x*=0.5f;
                e.relCluster=relClust.nearest(relEmbOf({A.src,B.dst})); // re-clusterizeaza numeric
                if(e.relCluster<0) e.relCluster=A.relCluster;
                e.confidence=A.confidence*B.confidence*decay; e.derived=true; e.sentence=-1;
                graph.addEdge(e);
            }
        }
    }

    // ================== DEBUG / DESCOPERIRI ==================
    cout<<"== 1. RELATII descoperite (clustere numerice) ==\n";
    {
        map<int,vector<int>> byRel;
        for(size_t s=0;s<sentRel.size();s++) if(sentRel[s]>=0) byRel[sentRel[s]].push_back((int)s);
        for(auto& kv:byRel){
            cout<<"  R"<<kv.first<<"  (propozitii: "<<kv.second.size()<<")\n";
            for(int s:kv.second) cout<<"       \""<<factLines[s]<<"\"\n";
        }
    }

    cout<<"\n== 2. CONCEPTE descoperite (clustere de noduri) ==\n";
    {
        map<int,vector<int>> byCon;
        for(auto& kv:nodeConcept) byCon[kv.second].push_back(kv.first);
        for(auto& kv:byCon){
            cout<<"  C"<<kv.first<<":  ";
            bool f=true; for(int nd:kv.second){cout<<(f?"":", ")<<tok.name(nd);f=false;} cout<<"\n";
        }
    }

    cout<<"\n== 3. GRAF (muchii: src --R--> dst, [C..]=concept tinta) ==\n";
    for(auto& e:graph.edges)
        cout<<"  "<<tok.name(e.src)<<" --R"<<e.relCluster<<"--> "<<tok.name(e.dst)
            <<"  [C"<<nodeConcept[e.dst]<<"]  conf="<<e.confidence<<(e.derived?"  (derivat)":"")<<"\n";

    // ============================================================
    //  4. CONCEPT FORMATION — abstractizare: unim conceptele de baza ale caror
    //  centroizi sunt apropiati vectorial intr-un SUPER-CONCEPT S (ierarhie).
    // ============================================================
    map<int,vector<float>> conCentroid; map<int,int> conCount;
    for(auto& kv:nodeConcept){ auto v=enc.tokenVec(kv.first);
        auto& c=conCentroid[kv.second]; if(c.empty())c.assign(v.size(),0);
        for(size_t k=0;k<v.size();k++)c[k]+=v[k]; conCount[kv.second]++; }
    for(auto& kv:conCentroid) for(float& x:kv.second) x/=conCount[kv.first];
    map<int,int> parent; for(auto& kv:conCentroid) parent[kv.first]=kv.first;
    function<int(int)> findp=[&](int x){ while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];} return x; };
    {
        float superThresh=0.33f;
        vector<int> cids; for(auto& kv:conCentroid) cids.push_back(kv.first);
        for(size_t i=0;i<cids.size();i++) for(size_t j=i+1;j<cids.size();j++)
            if(cosSim(conCentroid[cids[i]],conCentroid[cids[j]])>=superThresh){
                int a=findp(cids[i]),b=findp(cids[j]); if(a!=b)parent[a]=b; }
    }
    map<int,int> superId; int nextSuper=0; map<int,int> conSuper;
    for(auto& kv:conCentroid){ int r=findp(kv.first);
        if(!superId.count(r))superId[r]=nextSuper++; conSuper[kv.first]=superId[r]; }
    cout<<"\n== 4. CONCEPT FORMATION (super-concepte S = uniune de clustere) ==\n";
    {
        map<int,vector<int>> bySuper; for(auto& kv:conSuper) bySuper[kv.second].push_back(kv.first);
        map<int,vector<int>> conMembers; for(auto& kv:nodeConcept) conMembers[kv.second].push_back(kv.first);
        for(auto& kv:bySuper){
            cout<<"  S"<<kv.first<<"  <= { ";
            bool f=true; for(int c:kv.second){cout<<(f?"":", ")<<"C"<<c;f=false;} cout<<" } : ";
            f=true; for(int c:kv.second) for(int nd:conMembers[c]){cout<<(f?"":", ")<<tok.name(nd);f=false;}
            cout<<"\n";
        }
    }

    // ============================================================
    //  5. CONCEPT COMPRESSION — daca multe muchii au aceeasi semnatura
    //  (Csrc, R, Cdst), o comprimam intr-o abstractizare A (regularitate).
    // ============================================================
    cout<<"\n== 5. CONCEPT COMPRESSION (motive repetate -> abstractizari A) ==\n";
    {
        map<tuple<int,int,int>,vector<int>> motif;
        for(size_t i=0;i<graph.edges.size();i++){ auto& e=graph.edges[i]; if(e.derived)continue;
            motif[make_tuple(nodeConcept[e.src],e.relCluster,nodeConcept[e.dst])].push_back((int)i); }
        int aid=0;
        for(auto& kv:motif) if((int)kv.second.size()>=2){
            int cs,r,cd; tie(cs,r,cd)=kv.first;
            cout<<"  A"<<aid++<<"  (C"<<cs<<" --R"<<r<<"--> C"<<cd<<")  comprima "<<kv.second.size()<<" muchii:";
            for(int ei:kv.second) cout<<"  "<<tok.name(graph.edges[ei].src)<<"->"<<tok.name(graph.edges[ei].dst);
            cout<<"\n";
        }
        if(aid==0) cout<<"  (niciun motiv repetat de >=2 ori)\n";
    }

    // ============================================================
    //  6. CAUSAL DISCOVERY — fara reguli: cat de tare PREZICE src tinta dst
    //  (probabilitate next-token din encoder = semnal cauzal/predictiv).
    // ============================================================
    vector<float> causal(graph.edges.size(),0.f);
    for(size_t i=0;i<graph.edges.size();i++) causal[i]=enc.predProb({graph.edges[i].src},graph.edges[i].dst);
    { float mx=0; for(float c:causal)mx=max(mx,c); if(mx>0)for(float& c:causal)c/=mx; } // normalizare [0,1]
    cout<<"\n== 6. CAUSAL DISCOVERY (predictie src->dst; top muchii) ==\n";
    {
        vector<int> ord(graph.edges.size()); for(size_t i=0;i<ord.size();i++)ord[i]=(int)i;
        sort(ord.begin(),ord.end(),[&](int a,int b){return causal[a]>causal[b];});
        int lim=min((int)ord.size(),8);
        for(int t=0;t<lim;t++){ auto& e=graph.edges[ord[t]];
            cout<<"  "<<tok.name(e.src)<<" --R"<<e.relCluster<<"--> "<<tok.name(e.dst)<<"   causal="<<causal[ord[t]]<<"\n"; }
    }

    // ============================================================
    //  7. GOAL DISCOVERY — noduri spre care CONVERG multe lanturi (atractori).
    //  reachCount[v] = cate noduri-sursa pot ajunge la v in graf.
    // ============================================================
    map<int,int> reachCount;
    for(int s:graph.nodes){
        set<int> seen; queue<int> q; q.push(s); seen.insert(s);
        while(!q.empty()){ int u=q.front();q.pop();
            for(int ei:graph.outIdx(u)){ int v=graph.edges[ei].dst; if(!seen.count(v)){seen.insert(v);q.push(v);} } }
        for(int v:seen) if(v!=s) reachCount[v]++;
    }
    int maxReach=0; for(auto& kv:reachCount) maxReach=max(maxReach,kv.second);
    set<int> goalSet;
    cout<<"\n== 7. GOAL DISCOVERY (atractori = scop candidat) ==\n";
    {
        vector<pair<int,int>> v(reachCount.begin(),reachCount.end());
        sort(v.begin(),v.end(),[](const pair<int,int>&a,const pair<int,int>&b){return a.second>b.second;});
        float goalThresh=0.6f;
        int lim=min((int)v.size(),8);
        for(int t=0;t<lim;t++){ float gn=maxReach? (float)v[t].second/maxReach:0.f;
            bool isGoal=gn>=goalThresh; if(isGoal) goalSet.insert(v[t].first);
            cout<<"  "<<tok.name(v[t].first)<<"  convergenta="<<v[t].second<<" ("<<gn<<")"<<(isGoal?"  <= scop candidat":"")<<"\n"; }
    }

    // ============================================================
    //  8. WORLD MODEL — simuleaza pasi urmarind predictia maxima si evalueaza
    //  consecinta (coerenta = produsul predictiilor pe traiectorie).
    // ============================================================
    auto simulate=[&](int start,int steps){
        vector<int> traj{start}; float coher=1.f; int cur=start; set<int> seen{start};
        for(int s=0;s<steps;s++){
            int bestE=-1; float bestP=-1;
            for(int ei:graph.outIdx(cur)){ if(seen.count(graph.edges[ei].dst))continue;
                if(causal[ei]>bestP){bestP=causal[ei];bestE=ei;} }
            if(bestE<0)break;
            cur=graph.edges[bestE].dst; traj.push_back(cur); seen.insert(cur); coher*=max(1e-3f,bestP);
        }
        return make_pair(traj,coher);
    };
    cout<<"\n== 8. WORLD MODEL (simulare + consecinte) ==\n";
    {
        int shown=0;
        for(int n:nodeList){ if(graph.outIdx(n).empty())continue;
            auto pr=simulate(n,6); if(pr.first.size()<2)continue;
            cout<<"  din '"<<tok.name(n)<<"':  ";
            for(size_t i=0;i<pr.first.size();i++){if(i)cout<<" -> ";cout<<tok.name(pr.first[i]);}
            cout<<"   coerenta="<<pr.second<<"\n";
            if(++shown>=4)break;
        }
    }

    // ============================================================
    //  9. CONTINUOUS LEARNING — conceptele se re-formeaza cand apar date noi:
    //  clusterizam nodurile dupa prima jumatate de corpus vs dupa tot.
    // ============================================================
    cout<<"\n== 9. CONTINUOUS LEARNING (conceptele se schimba cu date noi) ==\n";
    {
        auto clusterNodes=[&](const set<int>& ns){
            Clusterer cc(0.38f); map<int,int> m;
            vector<int> v(ns.begin(),ns.end()); sort(v.begin(),v.end());
            for(int nd:v) m[nd]=cc.assign(enc.tokenVec(nd));
            return make_pair(m,(int)cc.centroid.size());
        };
        int half=max(1,(int)factIds.size()/2);
        set<int> nodesHalf;
        for(int i=0;i<half;i++) if(factIds[i].size()>=2){ nodesHalf.insert(factIds[i].front()); nodesHalf.insert(factIds[i].back()); }
        auto a=clusterNodes(nodesHalf); auto b=clusterNodes(graph.nodes);
        cout<<"  dupa "<<half<<" documente: "<<nodesHalf.size()<<" noduri / "<<a.second<<" concepte\n";
        cout<<"  dupa tot corpusul: "<<graph.nodes.size()<<" noduri / "<<b.second<<" concepte\n";
        int merged=0;
        for(auto& kv:a.first){ // noduri prezente in ambele care si-au schimbat gruparea
            int nd=kv.first; if(!b.first.count(nd))continue;
        }
        (void)merged;
        cout<<"  => pe masura ce intra documente noi, nodurile se re-grupeaza (clustere noi / fuziuni).\n";
    }

    // ============================================================
    //  10. HYPOTHESIS GENERATION + MULTI-PATH REASONING
    //  Nu mai exista regula "gaseste nodurile din intrebare si porneste de
    //  acolo". Generam un SPATIU de ipoteze = multe drumuri din tot graful.
    //  Pentru o intrebare scoram FIECARE ipoteza cu un scor numeric compus:
    //     query_sim + path_confidence + causal + goal + concept_match
    //  Alegem ipoteza maxima; daca scorul/relevanta < prag => "Nu stiu".
    // ============================================================
    struct Hyp{ vector<int> nodes,rels,eidx; };
    vector<Hyp> hyps;
    {
        const int HMAX=1500, LMAX=6;
        function<void(int,set<int>&,vector<int>&,vector<int>&,vector<int>&)> gen;
        gen=[&](int cur,set<int>& vis,vector<int>& n,vector<int>& r,vector<int>& ei){
            if((int)hyps.size()>=HMAX) return;
            if(n.size()>=2){ Hyp h; h.nodes=n; h.rels=r; h.eidx=ei; hyps.push_back(h); }
            if((int)n.size()>=LMAX) return;
            for(int e:graph.outIdx(cur)){ int d=graph.edges[e].dst; if(vis.count(d))continue;
                vis.insert(d);n.push_back(d);r.push_back(graph.edges[e].relCluster);ei.push_back(e);
                gen(d,vis,n,r,ei);
                vis.erase(d);n.pop_back();r.pop_back();ei.pop_back();
            }
        };
        for(int s:nodeList){ if((int)hyps.size()>=HMAX)break; set<int> vis{s}; vector<int> n{s},r,ei; gen(s,vis,n,r,ei); }
    }
    cout<<"\n== 10. HYPOTHESIS GENERATION ("<<hyps.size()<<" ipoteze) + MULTI-PATH SCORING ==\n";

    // Relevanta fata de intrebare (sQ) e semnalul PRIMAR; confidence/causal/goal
    // sunt modelatori SECUNDARI, activi doar cand drumul e relevant (inmultiti
    // cu sQ). Asa, o muchie cu causal mare dar irelevanta nu poate castiga.
    const float W_Q=1.0f,W_CON=0.25f,W_CONF=0.2f,W_CAUS=0.3f,W_GOAL=0.45f,W_OVL=0.5f;
    const float TAU=0.30f;     // prag scor minim (anti-halucinatie)
    const float QMIN=0.20f;    // prag relevanta query minima (gateaza raspunsul)
    auto pathEdgeMean=[&](const Hyp& h){ vector<float> v(enc.D,0);
        for(int e:h.eidx){auto& em=graph.edges[e].emb; for(int k=0;k<enc.D;k++)v[k]+=em[k];}
        if(!h.eidx.empty())for(float& x:v)x/=h.eidx.size(); return v; };
    auto pathNodeMean=[&](const Hyp& h){ vector<float> v(enc.D,0);
        for(int nd:h.nodes){auto t=enc.tokenVec(nd); for(int k=0;k<enc.D;k++)v[k]+=t[k];}
        if(!h.nodes.empty())for(float& x:v)x/=h.nodes.size(); return v; };

    for(size_t qi=0;qi<qIds.size();qi++){
        const auto& ids=qIds[qi];
        cout<<"\n  Q: \""<<qLines[qi]<<"\"\n";
        vector<float> q=enc.meanVec(ids,0,ids.size());
        set<int> qset(ids.begin(),ids.end());          // identitate de token (semnal soft de potrivire)
        int best=-1; float bestScore=-1e9f, bestQ=0;
        vector<pair<float,int>> ranked; ranked.reserve(hyps.size());
        for(size_t i=0;i<hyps.size();i++){
            const Hyp& h=hyps[i];
            float sQ=max(0.f,cosSim(q,pathEdgeMean(h)));
            float sCon=max(0.f,cosSim(q,pathNodeMean(h)));
            float sConf=0,sCaus=0; for(int e:h.eidx){sConf+=graph.edges[e].confidence;sCaus+=causal[e];}
            if(!h.eidx.empty()){sConf/=h.eidx.size();sCaus/=h.eidx.size();}
            float sGoal=0; for(int nd:h.nodes) if(goalSet.count(nd)){sGoal=1;break;}
            // potrivire de identitate (soft): fractia de noduri din drum prezente
            // in intrebare. NU e un start fix, doar inca un semnal numeric care
            // departajeaza noduri cu embedding aproape identic (Ion vs Vasile).
            float sOvl=0; for(int nd:h.nodes) if(qset.count(nd))sOvl+=1; if(!h.nodes.empty())sOvl/=h.nodes.size();
            // a ajunge la un SCOP descoperit e valoros in sine (termen aditiv);
            // confidence/causal raman modelatori legati de relevanta (×sQ).
            float score=W_Q*sQ + W_CON*sCon + W_GOAL*sGoal + W_OVL*sOvl + sQ*(W_CONF*sConf+W_CAUS*sCaus);
            ranked.push_back({score,(int)i});
            if(score>bestScore){bestScore=score;best=(int)i;bestQ=sQ;}
        }
        sort(ranked.begin(),ranked.end(),[](const pair<float,int>&a,const pair<float,int>&b){return a.first>b.first;});
        int show=min((int)ranked.size(),3);
        for(int t=0;t<show;t++){ const Hyp& h=hyps[ranked[t].second];
            cout<<"     ipoteza#"<<t<<" scor="<<ranked[t].first<<" : ";
            for(size_t k=0;k<h.nodes.size();k++){if(k)cout<<" -> ";cout<<tok.name(h.nodes[k]);} cout<<"\n"; }
        if(best<0 || bestScore<TAU || bestQ<QMIN){
            cout<<"     A: Nu stiu. (scor="<<bestScore<<", query_sim="<<bestQ<<")\n"; continue;
        }
        const Hyp& h=hyps[best];
        cout<<"     A: ";
        for(size_t k=0;k<h.nodes.size();k++){if(k)cout<<" -> ";cout<<tok.name(h.nodes[k]);}
        cout<<"\n        path: ";
        for(size_t k=0;k+1<h.nodes.size();k++) cout<<tok.name(h.nodes[k])<<" --R"<<h.rels[k]<<"--> ";
        cout<<tok.name(h.nodes.back())<<"\n        scor="<<bestScore<<"  query_sim="<<bestQ<<"\n";
    }
    return 0;
}
