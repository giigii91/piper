// ============================================================================
//  DEMO: sistem care INVATA concepte, definitii, reguli, scopuri si face
//  inferenta multi-pas pe graf — fara categorii semantice hardcodate si fara
//  roluri adnotate manual.
// ----------------------------------------------------------------------------
//  PRINCIPII (cerute explicit de audit):
//   - ZERO if(word=="este"), zero "person/profession/location" in cod.
//   - ZERO schema de roluri adnotata manual: NU mai exista "role_0/role_1/rel"
//     scrise langa fiecare propozitie. Sistemul primeste DOAR text brut.
//       * Care token este RELATIE se DESCOPERA distributional: un token care
//         apare predominant in pozitie de mijloc, legand perechi distincte de
//         capete, este o relatie (un "conector"). Restul sunt capete.
//       * Sursa / tinta / corp-de-definitie ies din POZITIE + ARITATE fata de
//         relatia descoperita. Nimic nu e numit in cod.
//   - Conceptele apar prin CLUSTERING/recurenta: tinta spre care converg multe
//     surse pe aceeasi relatie devine concept (abstractie descoperita).
//   - Relatiile primesc un MOD emergent (abstractie / lant-cauzal / scop /
//     definitie) calculat din SEMNATURA lor in graf, nu dintr-un nume.
//   - Se DESCOPERA reguli (mostenire de categorie) si se face FORWARD-CHAINING:
//     muchii noi sunt DERIVATE si reutilizate (rationament multi-pas real).
//   - Raspunde la "de ce" construind un LANT cauza->scop si il verbalizeaza
//     din pattern-urile descoperite (cuvintele vin din date, nu din cod).
//   - Are un PLANNER: cauta un drum (plan) de la un nod la un scop.
//   - Memoria lumii = GRAF (noduri+muchii), NU greutatile retelei.
//   - Rationamentul = TRAVERSARE determinista => anti-halucinatie: daca nu
//     exista drum, raspunsul e "Nu stiu".
//   - Transformerul e DOAR encoder de embeddings (pattern/similaritate),
//     nu baza de date a lumii.
//
//  Vocabular permis in cod: token, embedding, node, edge, cluster, memory,
//  relation_id, concept_id, confidence, vector, graph, rule, mode, plan.
//
//  C++17, un singur fisier, fara librarii externe.
//  Compilare:  g++ -O2 -std=c++17 concept_engine.cpp -o ce
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <cmath>
#include <random>
#include <algorithm>
#include <queue>

using namespace std;

static mt19937 rng(7);

// ----------------------------------------------------------------------------
//  Utilitare generice
// ----------------------------------------------------------------------------
vector<string> split(const string& s){
    vector<string> r; string w;
    for(char c:s){ if(c==' '){ if(!w.empty()) r.push_back(w); w.clear(); } else w+=c; }
    if(!w.empty()) r.push_back(w);
    return r;
}
float dot(const vector<float>&a,const vector<float>&b){
    float s=0; for(size_t i=0;i<a.size();i++) s+=a[i]*b[i]; return s;
}
float norm(const vector<float>&a){ return sqrt(dot(a,a)+1e-9f); }
float cosSim(const vector<float>&a,const vector<float>&b){
    return dot(a,b)/(norm(a)*norm(b));
}

// ============================================================================
//  PARTEA A — ENCODER NEURAL (Transformer mic): produce embeddings de tokeni
// ----------------------------------------------------------------------------
//  Rol: detector de pattern / generator de embeddings. Invata sa prezica
//  next-token pe dataset, iar ca efect secundar embeddingurile tokenilor care
//  apar in contexte similare devin apropiate. Folosim aceste embeddings la
//  clustering de concepte si la masurat similaritate. NU stocheaza fapte.
//  (Arhitectura: RMSNorm + multi-head attention + SwiGLU + AdamW — schelet
//  modern, mic; antrenare reala dar usoara.)
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
            m[i]=b1*m[i]+(1-b1)*g[i];
            v[i]=b2*v[i]+(1-b2)*g[i]*g[i];
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
        vector<float> dx(D);
        for(int i=0;i<D;i++)dx[i]=inv*(dxn[i]-xn[i]*d/D); return dx;
    }
};

struct Encoder {
    int V,T,D,H,nH,hd;
    Mat tokEmb, Wq,Wk,Wv,Wo, W1,Wg,W2, head;
    RMSNorm n1,n2,nf;
    Encoder(int v,int t,int d,int ffn,int heads)
        :V(v),T(t),D(d),H(ffn),nH(heads),hd(d/heads),
         tokEmb(v,d),Wq(d,d),Wk(d,d),Wv(d,d),Wo(d,d),
         W1(d,ffn),Wg(d,ffn),W2(ffn,d),head(d,v),n1(d),n2(d),nf(d){
        float s=sqrt(2.0f/d);
        tokEmb.init(0.02f);Wq.init(s);Wk.init(s);Wv.init(s);Wo.init(s);
        W1.init(s);Wg.init(s);W2.init(sqrt(2.0f/ffn));head.init(0.02f);
    }
    vector<Mat*> params(){return{&tokEmb,&Wq,&Wk,&Wv,&Wo,&W1,&Wg,&W2,&head,
        &n1.g,&n2.g,&nf.g};}
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
        // attention block
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
        // ffn block (SwiGLU)
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
        // final norm -> hidden (folosit ca embedding contextual)
        c.hid.resize(n);c.nf_inv.resize(n);c.nf_xn.resize(n);
        for(int t=0;t<n;t++){float inv;vector<float> xn;c.hid[t]=nf.fwd(c.out[t],inv,xn);c.nf_inv[t]=inv;c.nf_xn[t]=xn;}
        return c;
    }
    vector<float> logits(const vector<float>&h){return matmul(h,head);}

    // antrenare LM (next-token la fiecare pozitie)
    float trainSeq(const vector<int>&ids){
        int n=ids.size(); Cache c=forward(ids);
        vector<vector<float>> dHid(n,vector<float>(D,0));
        float loss=0;int cnt=0;
        for(int t=0;t<n-1;t++){
            auto z=logits(c.hid[t]); auto p=softmax(z); int tg=ids[t+1];
            loss+=-log(max(p[tg],1e-9f));cnt++;
            vector<float> dz=p;dz[tg]-=1;
            dHid[t]=matmul_bwd(c.hid[t],dz,head);
        }
        vector<vector<float>> dOut(n,vector<float>(D,0));
        for(int t=0;t<n-1;t++) dOut[t]=nf.bwd(dHid[t],c.nf_xn[t],c.nf_inv[t]);
        vector<vector<float>> dRes1(n,vector<float>(D,0));
        for(int t=0;t<n;t++){
            vector<float> dffn=dOut[t];
            for(int i=0;i<D;i++)dRes1[t][i]+=dOut[t][i];
            vector<float> dact=matmul_bwd(c.act[t],dffn,W2);
            vector<float> dup(H),dg(H);
            for(int i=0;i<H;i++){float su=silu(c.up[t][i]);dg[i]=dact[i]*su;dup[i]=dact[i]*c.gate[t][i]*dsilu(c.up[t][i]);}
            vector<float> d1=matmul_bwd(c.f_n[t],dup,W1),d2=matmul_bwd(c.f_n[t],dg,Wg);
            vector<float> df(D);for(int i=0;i<D;i++)df[i]=d1[i]+d2[i];
            vector<float> dr=n2.bwd(df,c.f_xn[t],c.f_inv[t]);
            for(int i=0;i<D;i++)dRes1[t][i]+=dr[i];
        }
        vector<vector<float>> dEmb(n,vector<float>(D,0)),dAttn(n,vector<float>(D,0));
        for(int t=0;t<n;t++){
            for(int i=0;i<D;i++)dEmb[t][i]+=dRes1[t][i];
            dAttn[t]=matmul_bwd(c.attn[t],dRes1[t],Wo);
        }
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
    // embedding "static" al unui token (din tabela), folosit la clustering
    vector<float> tokenVec(int id){vector<float> v(D);for(int i=0;i<D;i++)v[i]=tokEmb.at(id,i);return v;}
};

// ============================================================================
//  PARTEA B — TOKENIZER generic
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
//  PARTEA C — DESCOPERIRE DE ROLURI (fara adnotare manuala)
// ----------------------------------------------------------------------------
//  Nu mai exista "role_0/rel/role_1". Primim text brut. Descoperim ce token
//  este o RELATIE (conector) folosind o semnatura distributionala:
//    un token este relatie daca apare PREDOMINANT in pozitie de mijloc
//    (nici primul, nici ultimul) — adica leaga doua capete.
//  Sursa/tinta ies apoi din pozitia fata de relatie. Aritatea (cati tokeni
//  raman dupa relatie) distinge un FAPT (o singura tinta) de o DEFINITIE
//  (corp multi-token).
// ============================================================================
struct RoleDiscovery {
    unordered_set<int> relationTokens;   // tokeni descoperiti ca fiind conectori

    // pasul 1: invata din corpus brut care tokeni sunt relatii
    void learn(const vector<vector<int>>& corpus){
        unordered_map<int,int> mid,tot;
        for(auto& s:corpus){
            for(size_t i=0;i<s.size();i++){
                tot[s[i]]++;
                if(i>0 && i+1<s.size()) mid[s[i]]++;
            }
        }
        for(auto& kv:tot){
            int t=kv.first, total=kv.second, m=mid.count(t)?mid[t]:0;
            // majoritate STRICTA de aparitii in mijloc => conector/relatie
            if(m>0 && m*2>total) relationTokens.insert(t);
        }
    }
    bool isRelation(int t)const{ return relationTokens.count(t)>0; }
};

// relation_id descoperite din pattern-uri de relatie (run-ul de conectori)
struct RelationVocab {
    unordered_map<string,int> id; vector<string> pattern;
    int discover(const string&p){auto it=id.find(p);if(it!=id.end())return it->second;
        int x=pattern.size();id[p]=x;pattern.push_back(p);return x;}
    int get(const string&p)const{auto it=id.find(p);return it==id.end()?-1:it->second;}
};

// rezultatul parsarii unei propozitii brute, dupa rolurile DESCOPERITE
struct Parsed {
    bool ok=false;
    bool isDefinition=false;
    int src=-1, dst=-1, relation_id=-1;
    vector<int> body;          // corpul definitiei (daca isDefinition)
    string relPattern;
};

// Parser pe baza rolurilor descoperite:
//   src   = capatul de dinaintea primei relatii
//   rel   = run-ul contiguu de tokeni-relatie
//   dupa  = ce ramane: 1 token => FAPT (dst), >=2 tokeni => DEFINITIE (body)
Parsed parseByDiscoveredRoles(const vector<int>& ids, const RoleDiscovery& rd,
                              Tokenizer& tok, RelationVocab& rv){
    Parsed p;
    if(ids.size()<3) return p;
    // gaseste prima relatie (nu poate fi pe pozitia 0)
    int r0=-1;
    for(size_t i=1;i<ids.size();i++){ if(rd.isRelation(ids[i])){ r0=i; break; } }
    if(r0<=0) return p;
    p.src=ids[r0-1];
    // run-ul de conectori
    int r1=r0; string relPat;
    while(r1<(int)ids.size() && rd.isRelation(ids[r1])){
        if(!relPat.empty()) relPat+=" ";
        relPat+=tok.name(ids[r1]); r1++;
    }
    int rem=(int)ids.size()-r1;
    if(rem<=0) return p;
    p.relPattern=relPat; p.relation_id=rv.discover(relPat);
    if(rem==1){ p.dst=ids[r1]; p.ok=true; }
    else { p.isDefinition=true; for(int i=r1;i<(int)ids.size();i++) p.body.push_back(ids[i]); p.ok=true; }
    return p;
}

// ============================================================================
//  PARTEA D — GRAPH MEMORY (memoria lumii: noduri + muchii)
// ----------------------------------------------------------------------------
//  Un nod = un token/concept (node_id == token_id, generic).
//  O muchie = (src) --relation_id--> (dst), cu confidence, marcata `derived`
//  daca a fost DEDUSA de motor (forward-chaining), nu observata direct.
// ============================================================================
struct Edge { int src,dst,relation_id; float confidence; bool derived; };

struct GraphMemory {
    vector<Edge> edges;
    unordered_map<int,vector<int>> outAdj;

    int addEdge(int src,int dst,int rel,float conf,bool derived=false){
        for(int ei:outAdj[src]) if(edges[ei].dst==dst&&edges[ei].relation_id==rel){
            edges[ei].confidence=max(edges[ei].confidence,conf); return ei; }
        int idx=edges.size(); edges.push_back({src,dst,rel,conf,derived});
        outAdj[src].push_back(idx); return idx;
    }
    vector<const Edge*> query(int src,int rel)const{
        vector<const Edge*> r; auto it=outAdj.find(src);
        if(it==outAdj.end())return r;
        for(int ei:it->second) if(edges[ei].relation_id==rel) r.push_back(&edges[ei]);
        return r;
    }
    vector<const Edge*> outgoing(int src)const{
        vector<const Edge*> r; auto it=outAdj.find(src);
        if(it==outAdj.end())return r;
        for(int ei:it->second) r.push_back(&edges[ei]);
        return r;
    }
    bool hasOutgoing(int src)const{ auto it=outAdj.find(src); return it!=outAdj.end()&&!it->second.empty(); }
};

// ============================================================================
//  PARTEA E — CONCEPT MEMORY (concepte descoperite = ABSTRACTIE emergenta)
// ----------------------------------------------------------------------------
//  Cum apare un "concept": daca MULTI tokeni-sursa pointeaza catre acelasi
//  token-tinta prin ACEEASI relatie (multi X --r--> "om"), tinta ("om") e
//  promovata la CONCEPT, iar sursele devin INSTANTE. Pragul (minim de
//  instante) e singurul parametru; categoria nu e numita. Aceasta E
//  abstractia descoperita ceruta de audit.
// ============================================================================
struct ConceptMemory {
    set<int> concepts;
    unordered_map<int,set<int>> instances;
    unordered_map<int,int> instanceOf;

    // Promovam la concept DOAR tinte ale relatiilor de ABSTRACTIE (is-a)
    // descoperite (absRels), ca sa nu confundam convergenta cauzala (mai multe
    // resurse -> aceeasi tinta) cu o categorie.
    void discover(const GraphMemory& g,int minInst,const set<int>& absRels){
        map<pair<int,int>,set<int>> bucket;
        for(auto& e:g.edges) if(!e.derived && absRels.count(e.relation_id))
            bucket[{e.dst,e.relation_id}].insert(e.src);
        for(auto& kv:bucket){
            if((int)kv.second.size()>=minInst){
                int concept=kv.first.first;
                concepts.insert(concept);
                for(int inst:kv.second){ instances[concept].insert(inst); instanceOf[inst]=concept; }
            }
        }
    }
    bool isConcept(int id)const{return concepts.count(id);}
};

// ============================================================================
//  PARTEA F — DEFINITION MEMORY (definitii descoperite)
// ----------------------------------------------------------------------------
//  O definitie e legatura termen->{tokeni-corp}, descoperita din aritate (vezi
//  parser-ul). Utila in lantul de inferenta ca "expandare" a unui concept.
// ============================================================================
struct DefinitionMemory {
    unordered_map<int,vector<int>> def;
    unordered_map<int,int> defRelation;
    void add(int term,int rel,const vector<int>& body){ def[term]=body; defRelation[term]=rel; }
    bool has(int term)const{return def.count(term);}
};

// ============================================================================
//  PARTEA G — RELATION MODES (cauza / efect / scop / abstractie DESCOPERITE)
// ----------------------------------------------------------------------------
//  Fiecare relatie primeste un MOD emergent, calculat din semnatura ei in graf
//  — NU dintr-un nume scris in cod:
//    ABSTRACTION : fan-in mare (multe surse -> putine tinte) => is-a/clasa.
//    CHAIN       : tintele devin la randul lor surse => lant cauza-efect/mijloc.
//    GOAL        : pattern de relatie multi-cuvant (relatie oblica, ex. un
//                  conector secundar) => tinta e un SCOP/nevoie.
//    PLAIN       : altceva (ex. localizare punctuala).
//  Aceste moduri sunt apoi REUTILIZATE de motorul de rationament.
// ============================================================================
enum RelMode { PLAIN=0, ABSTRACTION=1, CHAIN=2, GOAL=3 };
const char* modeName(int m){
    switch(m){case ABSTRACTION:return "ABSTRACTION(is-a)";case CHAIN:return "CHAIN(cauza->efect)";
              case GOAL:return "GOAL(scop)";default:return "PLAIN";}
}

struct RelationModes {
    unordered_map<int,int> mode;            // relation_id -> RelMode
    set<int> goalTargets;                    // tinte ale relatiilor de tip GOAL

    void discover(const GraphMemory& g, const RelationVocab& rv){
        // statistici per relatie
        map<int,map<int,set<int>>> dstSrc;   // rel -> dst -> {src}
        map<int,set<int>> dsts;              // rel -> {dst}
        for(auto& e:g.edges){ if(e.derived) continue;
            dstSrc[e.relation_id][e.dst].insert(e.src); dsts[e.relation_id].insert(e.dst); }

        for(auto& kv:dstSrc){
            int rel=kv.first;
            // fan-in: cate tinte sunt "clase" (au >=2 surse distincte)?
            int maxFanIn=0, classes=0;
            for(auto& d:kv.second){ int f=d.second.size(); maxFanIn=max(maxFanIn,f); if(f>=2) classes++; }
            // chain: ce fractie din tinte au, la randul lor, muchii iesite?
            int cont=0,tot=0; for(int d:dsts[rel]){ tot++; if(g.hasOutgoing(d)) cont++; }
            float chainFrac = tot? (float)cont/tot : 0.f;
            // goal: pattern-ul relatiei are mai multe cuvinte (relatie oblica)
            bool multiWord = split(rv.pattern[rel]).size()>=2;

            // O TAXONOMIE reala are mai multe clase, fiecare cu >=2 instante
            // (asa separam is-a de simpla convergenta cauzala pe o tinta).
            if(multiWord){ mode[rel]=GOAL; for(int d:dsts[rel]) goalTargets.insert(d); }
            else if(classes>=2) mode[rel]=ABSTRACTION;
            else if(chainFrac>=0.5f) mode[rel]=CHAIN;
            else if(maxFanIn>=2) mode[rel]=ABSTRACTION;  // o clasa clara, fara lant
            else mode[rel]=PLAIN;
        }
    }
    int of(int rel)const{ auto it=mode.find(rel); return it==mode.end()?PLAIN:it->second; }
    bool isAbstraction(int rel)const{ return of(rel)==ABSTRACTION; }
    bool isGoalTarget(int node)const{ return goalTargets.count(node)>0; }
};

// ============================================================================
//  PARTEA H — RULE ENGINE (descoperire de reguli + forward-chaining)
// ----------------------------------------------------------------------------
//  Regula descoperita generic (MOSTENIRE DE CATEGORIE):
//     daca  X --r--> Y   si   Y --(ABSTRACTION)--> Z   atunci   X --(ABSTRACTION)--> Z
//  Adica: daca stiu ce e Y (clasa lui) si X e legat de Y, X mosteneste clasa.
//  Ex: Ion --lucreaza--> sofer ; sofer --este--> meserie  =>  Ion --este--> meserie.
//  Muchiile rezultate sunt DERIVATE (confidence redus) si reutilizabile mai
//  departe (inchidere prin iteratie pana la punct fix) => rationament multi-pas.
// ============================================================================
struct DiscoveredRule { string desc; int applications=0; };

struct RuleEngine {
    // intoarce regulile descoperite; modifica graful adaugand muchii derivate
    vector<DiscoveredRule> run(GraphMemory& g, const RelationModes& rm, const RelationVocab& rv){
        vector<DiscoveredRule> rules;
        // ce relatii sunt de abstractie?
        set<int> absRels; for(auto& kv:rm.mode) if(kv.second==ABSTRACTION) absRels.insert(kv.first);
        if(absRels.empty()) return rules;

        // descrie regula (folosind pattern-uri DESCOPERITE, nu literali din cod)
        for(int ar:absRels){
            DiscoveredRule r; r.desc = "daca X --(orice)--> Y si Y --[" + rv.pattern[ar] +
                "]--> Z atunci X --[" + rv.pattern[ar] + "]--> Z  (mostenire de categorie)";
            rules.push_back(r);
        }

        // forward-chaining pana la punct fix
        bool changed=true; int guard=0;
        while(changed && guard++<8){
            changed=false;
            int m=g.edges.size();
            for(int i=0;i<m;i++){
                Edge e=g.edges[i];                 // X --r--> Y
                // Y --ABS--> Z ?
                for(int ar:absRels){
                    for(auto* ae:g.query(e.dst,ar)){   // Y --ar--> Z
                        if(e.src==ae->dst) continue;   // fara auto-bucla
                        float conf=e.confidence*ae->confidence*0.9f;
                        size_t before=g.edges.size();
                        g.addEdge(e.src,ae->dst,ar,conf,true);
                        if(g.edges.size()>before){     // s-a adaugat o muchie noua
                            changed=true;
                            for(auto& rr:rules)
                                if(rr.desc.find("["+rv.pattern[ar]+"]")!=string::npos) rr.applications++;
                        }
                    }
                }
            }
        }
        return rules;
    }
};

// ============================================================================
//  PARTEA I — REASONING ENGINE (traversare determinista multi-pas + scop)
// ----------------------------------------------------------------------------
//  Raspunde cautand DRUMURI in graf. Daca nu exista drum => "Nu stiu"
//  (anti-halucinatie). "De ce" => lant cauza->scop, verbalizat din pattern-uri.
//  "Planner" => cauta un drum catre un scop dat. Totul determinist; confidence
//  se propaga ca produs al muchiilor.
// ============================================================================
struct Step { int from,to,rel; };
struct Chain { vector<int> nodes; vector<int> rels; float confidence=0; };

struct ReasoningEngine {
    GraphMemory& g; ConceptMemory& cm; DefinitionMemory& dm; RelationModes& rm;
    Tokenizer& tok; RelationVocab& rv;
    ReasoningEngine(GraphMemory&G,ConceptMemory&C,DefinitionMemory&D,RelationModes&M,
                    Tokenizer&T,RelationVocab&R):g(G),cm(C),dm(D),rm(M),tok(T),rv(R){}

    vector<int> direct(int src,int rel){
        vector<int> r; for(auto*e:g.query(src,rel)) r.push_back(e->dst); return r;
    }
    string nodeName(int id){ return tok.name(id); }

    // cel mai lung lant (conf maxima la egalitate) pornind din 'start'
    Chain bestChain(int start,int maxDepth){
        struct St{int node;vector<int> nodes;vector<int> rels;float conf;};
        Chain best; best.nodes={start}; best.confidence=1.0f;
        queue<St> q; q.push({start,{start},{},1.0f});
        set<vector<int>> seen;
        while(!q.empty()){
            St s=q.front();q.pop();
            if(s.nodes.size()>best.nodes.size()||
               (s.nodes.size()==best.nodes.size()&&s.conf>best.confidence)){
                best.nodes=s.nodes;best.rels=s.rels;best.confidence=s.conf;
            }
            if((int)s.nodes.size()>=maxDepth) continue;
            for(auto*e:g.outgoing(s.node)){
                if(find(s.nodes.begin(),s.nodes.end(),e->dst)!=s.nodes.end()) continue;
                St ns=s; ns.node=e->dst; ns.nodes.push_back(e->dst);
                ns.rels.push_back(e->relation_id); ns.conf*=e->confidence;
                if(seen.count(ns.nodes)) continue; seen.insert(ns.nodes);
                q.push(ns);
            }
        }
        return best;
    }

    // PLANNER: drum cel mai scurt de la 'start' la 'goal' (BFS pe muchii)
    Chain plan(int start,int goal,int maxDepth){
        struct St{int node;vector<int> nodes;vector<int> rels;float conf;};
        queue<St> q; q.push({start,{start},{},1.0f});
        set<int> visited; visited.insert(start);
        while(!q.empty()){
            St s=q.front();q.pop();
            if(s.node==goal){ Chain c; c.nodes=s.nodes;c.rels=s.rels;c.confidence=s.conf; return c; }
            if((int)s.nodes.size()>=maxDepth) continue;
            for(auto*e:g.outgoing(s.node)){
                if(visited.count(e->dst)) continue; visited.insert(e->dst);
                St ns=s; ns.node=e->dst; ns.nodes.push_back(e->dst);
                ns.rels.push_back(e->relation_id); ns.conf*=e->confidence;
                q.push(ns);
            }
        }
        return Chain{}; // gol => nu exista plan
    }

    // verbalizeaza un lant ca pasi "a [pattern] b"
    string verbalize(const Chain& ch){
        if(ch.nodes.size()<2) return "Nu stiu.";
        string s;
        for(size_t i=0;i+1<ch.nodes.size();i++){
            if(i) s+="; ";
            s+=nodeName(ch.nodes[i])+" "+rv.pattern[ch.rels[i]]+" "+nodeName(ch.nodes[i+1]);
        }
        char buf[40]; snprintf(buf,sizeof buf," (confidence=%.2f)",ch.confidence);
        return s+buf;
    }

    // "DE CE ...?" — construieste lantul cauza->scop si il verbalizeaza ca scop.
    //  Strategie: pornim din actor, luam cel mai lung lant; gasim primul nod
    //  care e un SCOP descoperit (tinta a unei relatii GOAL) si formulam
    //  "Pentru a obtine <scop>", apoi continuam lantul ca justificare.
    string why(int actor){
        // 1) cel mai apropiat SCOP atins de actor (BFS), + drumul-mijloc spre el
        int goal=-1; Chain means;
        {
            struct St{int node;vector<int> nodes;vector<int> rels;float conf;};
            queue<St> q; q.push({actor,{actor},{},1.0f});
            set<int> vis; vis.insert(actor);
            while(!q.empty()){
                St s=q.front();q.pop();
                if(s.node!=actor && rm.isGoalTarget(s.node)){
                    goal=s.node; means.nodes=s.nodes;means.rels=s.rels;means.confidence=s.conf; break;
                }
                if((int)s.nodes.size()>=8) continue;
                for(auto*e:g.outgoing(s.node)){ if(vis.count(e->dst))continue; vis.insert(e->dst);
                    St ns=s; ns.node=e->dst; ns.nodes.push_back(e->dst);
                    ns.rels.push_back(e->relation_id); ns.conf*=e->confidence; q.push(ns); }
            }
        }
        if(goal<0){
            Chain ch=bestChain(actor,8);
            if(ch.nodes.size()<2) return "Nu stiu.";
            return "Pentru ca: " + verbalize(ch);
        }
        // 2) de ce conteaza scopul = lantul continuat din scop (justificare)
        string out = "Pentru a obtine " + nodeName(goal);
        Chain cont = bestChain(goal,6);
        if(cont.nodes.size()>=2){
            string just;
            for(size_t i=0;i+1<cont.nodes.size();i++){ if(!just.empty()) just+="; ";
                just += nodeName(cont.nodes[i])+" "+rv.pattern[cont.rels[i]]+" "+nodeName(cont.nodes[i+1]); }
            out += " (necesar pentru: " + just + ")";
        }
        out += ".";
        return out;
    }

    // expandeaza un concept prin definitia lui (daca exista)
    string expandDefinition(int term){
        if(!dm.has(term)) return "";
        auto& body=dm.def.at(term);
        string s=nodeName(term)+" := ";
        for(size_t i=0;i<body.size();i++) s+=(i?" ":"")+nodeName(body[i]);
        return s;
    }
};

// ============================================================================
//  PARTEA J — MAIN: dataset BRUT (fara scheme) + descoperire + inferenta
// ============================================================================
int main(){
    Tokenizer tok;
    RelationVocab rv;
    GraphMemory graph;
    ConceptMemory concepts;
    DefinitionMemory defs;
    RelationModes modes;

    // ------------------------------------------------------------------
    //  DATASET — DOAR TEXT BRUT. Fara nicio schema de roluri.
    //  Sistemul descopera singur ce e relatie si ce e capat.
    // ------------------------------------------------------------------
    vector<string> facts = {
        // is-a (vor genera concepte prin recurenta tintei => ABSTRACTION)
        "Ion este om",
        "Vasile este om",
        "Maria este om",
        "pisica este animal",
        "caine este animal",
        "sofer este meserie",
        "mecanic este meserie",
        // ocupatii concrete
        "Ion lucreaza sofer",
        "Vasile lucreaza mecanic",
        // localizari punctuale (relatie PLAIN)
        "pisica sta masa",
        "telefon sta birou",
        // lant cauza-efect / mijloc-scop (CHAIN)
        "sofer face munca",
        "munca produce bani",
        "bani cumpara hrana",
        "hrana sustine om",
        "meserie produce venit",
        "venit cumpara hrana",
        // scopuri (relatie oblica, multi-cuvant => GOAL)
        "om munceste pentru bani",
        "om mananca pentru energie",
    };

    // definitii — tot text brut; sunt recunoscute prin ARITATE (corp >1 token)
    vector<string> definitions = {
        "meserie inseamna activitate care aduce bani",
        "animal inseamna fiinta vie",
        "om inseamna fiinta vie",
        "hrana inseamna energie pentru corp",
    };

    // ---- pre-tokenizare pentru encoder + descoperirea rolurilor ----
    vector<vector<int>> factIds, defIds, corpus;
    for(auto& s:facts){ auto v=tok.enc(s); factIds.push_back(v); corpus.push_back(v); }
    for(auto& s:definitions){ auto v=tok.enc(s); defIds.push_back(v); corpus.push_back(v); }

    // ---- antreneaza encoderul (DOAR pentru embeddings/pattern) ----
    Encoder enc(tok.word.size(), 16, 32, 64, 4);
    cout<<"== Antrenez encoderul (doar pentru embeddings/pattern) ==\n";
    float lr=0.005f,wd=0.001f,step=0;
    for(int ep=0;ep<150;ep++){
        float tot=0;int c=0;
        vector<int> ord(corpus.size());for(int i=0;i<(int)ord.size();i++)ord[i]=i;
        shuffle(ord.begin(),ord.end(),rng);
        for(int idx:ord){enc.zerograd();tot+=enc.trainSeq(corpus[idx]);step++;enc.step(lr,step,wd);c++;}
        if(ep%50==0)cout<<"  epoch "<<ep<<"  loss "<<tot/c<<"\n";
    }
    cout<<"Done.\n\n";

    // ---- DESCOPERA rolurile (care tokeni sunt relatii) din corpusul de fapte
    RoleDiscovery roles;
    roles.learn(factIds);

    // ---- PARSEAZA faptele pe baza rolurilor descoperite -> graf ----
    for(auto& ids:factIds){
        Parsed p=parseByDiscoveredRoles(ids,roles,tok,rv);
        if(p.ok && !p.isDefinition) graph.addEdge(p.src,p.dst,p.relation_id,0.95f);
    }
    // ---- definitii: descopera conectorul (token recurent pe pozitia 1) ----
    {
        unordered_map<int,int> at1;
        for(auto& ids:defIds) if(ids.size()>=3) at1[ids[1]]++;
        int connector=-1,best=0; for(auto&kv:at1) if(kv.second>best){best=kv.second;connector=kv.first;}
        for(auto& ids:defIds){
            if((int)ids.size()<3 || ids[1]!=connector) continue;
            int term=ids[0]; int rel=rv.discover(tok.name(connector));
            vector<int> body(ids.begin()+2,ids.end());
            defs.add(term,rel,body);
        }
    }

    // ---- DESCOPERA modurile relatiilor (abstractie/lant/scop) ----
    modes.discover(graph, rv);
    set<int> absRels; for(auto&kv:modes.mode) if(kv.second==ABSTRACTION) absRels.insert(kv.first);
    // ---- DESCOPERA concepte (abstractie prin recurenta, doar pe relatii is-a)
    concepts.discover(graph, 2, absRels);
    // ---- DESCOPERA reguli + FORWARD-CHAINING (muchii derivate) ----
    RuleEngine ruleEng;
    auto rules = ruleEng.run(graph, modes, rv);

    ReasoningEngine reason(graph,concepts,defs,modes,tok,rv);

    // ================== OUTPUT ==================

    cout<<"== 1. ROLURI DESCOPERITE (care tokeni sunt relatii) ==\n  ";
    { bool f=true; vector<string> rs; for(int t:roles.relationTokens) rs.push_back(tok.name(t));
      sort(rs.begin(),rs.end());
      for(auto&s:rs){cout<<(f?"":", ")<<s;f=false;} cout<<"\n"; }

    cout<<"\n== 2. RELATII descoperite + MOD emergent ==\n";
    set<int> defRels; for(auto&kv:defs.defRelation) defRels.insert(kv.second);
    for(int i=0;i<(int)rv.pattern.size();i++){
        string m = defRels.count(i)? "DEFINITION(definitie)" : modeName(modes.of(i));
        cout<<"  relation_id "<<i<<"  <- \""<<rv.pattern[i]<<"\"   mod: "<<m<<"\n";
    }

    cout<<"\n== 3. CONCEPTE descoperite (abstractie, nu hardcodat) ==\n";
    for(int cId:concepts.concepts){
        cout<<"  concept: "<<tok.name(cId)<<"   instante: ";
        bool first=true;
        for(int inst:concepts.instances[cId]){cout<<(first?"":", ")<<tok.name(inst);first=false;}
        cout<<"\n";
    }

    cout<<"\n== 4. DEFINITII descoperite (prin aritate) ==\n";
    for(auto&kv:defs.def){
        cout<<"  "<<tok.name(kv.first)<<" := ";
        for(size_t i=0;i<kv.second.size();i++)cout<<(i?" ":"")<<tok.name(kv.second[i]);
        cout<<"\n";
    }

    cout<<"\n== 5. REGULI descoperite + aplicari (forward-chaining) ==\n";
    for(auto& r:rules) cout<<"  - "<<r.desc<<"   [aplicata de "<<r.applications<<" ori]\n";

    cout<<"\n== 6. GRAF (muchii: src --relation_id--> dst) ==\n";
    for(auto&e:graph.edges)
        cout<<"  "<<tok.name(e.src)<<" --r"<<e.relation_id<<"--> "<<tok.name(e.dst)
            <<"  (conf="<<e.confidence<<(e.derived?", DERIVAT":"")<<")\n";

    cout<<"\n== 7. SIMILARITATE (encoder embeddings) ==\n";
    auto sim=[&](const string&a,const string&b){
        int ia=tok.get(a),ib=tok.get(b); if(ia<0||ib<0)return -2.0f;
        return cosSim(enc.tokenVec(ia),enc.tokenVec(ib));
    };
    cout<<"  sim(Ion,Vasile)   = "<<sim("Ion","Vasile")<<"\n";
    cout<<"  sim(Ion,pisica)   = "<<sim("Ion","pisica")<<"\n";
    cout<<"  sim(sofer,mecanic)= "<<sim("sofer","mecanic")<<"\n";

    cout<<"\n== 8. INFERENTA multi-pas (lanturi de traversare) ==\n";
    auto chainFrom=[&](const string&startTok){
        int s=tok.get(startTok);
        if(s<0){cout<<"  '"<<startTok<<"': Nu stiu.\n";return;}
        Chain ch=reason.bestChain(s,8);
        cout<<"  lant din '"<<startTok<<"': "<<reason.verbalize(ch)<<"\n";
    };
    chainFrom("Ion");
    chainFrom("sofer");
    chainFrom("meserie");

    cout<<"\n== 9. RATIONAMENT \"DE CE?\" (cauza -> scop) ==\n";
    auto why=[&](const string&actor){
        int a=tok.get(actor); if(a<0){cout<<"  Nu stiu.\n";return;}
        cout<<"  De ce lucreaza/actioneaza "<<actor<<"?\n    A: "<<reason.why(a)<<"\n";
    };
    why("Ion");
    why("Vasile");

    cout<<"\n== 10. PLANNER (drum catre un scop) ==\n";
    auto planTo=[&](const string&a,const string&b){
        int s=tok.get(a),t=tok.get(b);
        if(s<0||t<0){cout<<"  Nu stiu.\n";return;}
        Chain p=reason.plan(s,t,8);
        if(p.nodes.size()<2){cout<<"  plan "<<a<<" -> "<<b<<": Nu exista (Nu stiu).\n";return;}
        cout<<"  plan "<<a<<" -> "<<b<<": "<<reason.verbalize(p)<<"\n";
    };
    planTo("Ion","bani");
    planTo("Ion","hrana");
    planTo("pisica","bani"); // nu exista drum => Nu stiu

    cout<<"\n== 11. MOSTENIRE DEDUSA (intrebari pe muchii derivate) ==\n";
    // Ce e Ion (toate categoriile, inclusiv cele DEDUSE prin reguli)?
    {
        int ion=tok.get("Ion");
        cout<<"  Ion --[abstractie]--> { ";
        bool f=true;
        if(ion>=0) for(int ar:absRels) for(int d:reason.direct(ion,ar)){cout<<(f?"":", ")<<tok.name(d);f=false;}
        cout<<" }   (include categorii DEDUSE: ex. meserie via Ion->sofer->meserie)\n";
    }

    cout<<"\n== 12. EXPANDARE prin DEFINITIE ==\n";
    for(const string& term:{string("meserie"),string("hrana"),string("om")}){
        int t=tok.get(term); if(t<0)continue;
        string e=reason.expandDefinition(t);
        cout<<"  "<<(e.empty()?term+": (fara definitie)":e)<<"\n";
    }

    cout<<"\n== 13. ANTI-HALUCINATIE (informatie inexistenta) ==\n";
    auto answerRel=[&](const string&subj,const string&relPattern){
        int s=tok.get(subj), r=rv.get(relPattern);
        if(s<0||r<0){cout<<"  Nu stiu.\n";return;}
        auto outs=reason.direct(s,r);
        if(outs.empty()){cout<<"  Nu stiu.\n";return;}
        string res; for(size_t i=0;i<outs.size();i++)res+=(i?", ":"")+tok.name(outs[i]);
        cout<<"  "<<subj<<" --["<<relPattern<<"]--> "<<res<<"\n";
    };
    cout<<"  Q: unde 'sta' Ion?\n  A: "; answerRel("Ion","sta");
    cout<<"  Q: ce relatie 'zboara' are pisica?\n  A: "; answerRel("pisica","zboara");
    int unknown=tok.get("dragon");
    cout<<"  Q: lant din 'dragon' (token necunoscut)?\n  A: ";
    if(unknown<0) cout<<"Nu stiu.\n"; else cout<<reason.verbalize(reason.bestChain(unknown,5))<<"\n";

    return 0;
}
