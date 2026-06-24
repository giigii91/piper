// ============================================================================
//  DEMO: sistem neuro-simbolic care INVATA PRIN CITIRE (ca un LLM care digera
//  text), construieste o GEOMETRIE de graf si GENEREAZA prin "Next-Token pe
//  graf" — traversare ghidata de vectorul de context al transformerului.
// ----------------------------------------------------------------------------
//  PRINCIPII (cerute explicit de audit):
//   - FARA TEXT BRUT IN COD: cunostintele NU sunt string-uri scrise in sursa.
//     Motorul CITESTE un corpus extern (knowledge.txt / argv[1] / stdin),
//     exact ca un om care citeste un manual, si il transforma in noduri+muchii.
//     Codul nu contine propozitii-cunostinta si nici entitati hardcodate:
//     exemplele de la rulare sunt ALESE automat din structurile descoperite.
//   - ROLURI DESCOPERITE: care token e relatie se afla distributional
//     (pozitie de mijloc). Sursa/tinta/corp ies din pozitie + aritate.
//   - MODURI emergente de relatie (abstractie / lant-cauzal / scop / definitie)
//     calculate din semnatura in graf, nu din nume.
//   - REGULI descoperite + FORWARD-CHAINING (muchii derivate, reutilizate).
//   - "NEXT-TOKEN PE GRAF" (generativ): urmatorul token NU iese dintr-o matrice
//     de greutati, ci e ALES dintre VECINII fizici din graf; transformerul mic
//     doar SCORzeaza candidatii dupa similaritatea cu vectorul de context.
//       * mod determinist (logica/programare): doar muchii reale, conf maxima.
//       * mod creativ (poezie/metafora): cand nu exista muchie, "sare" la cel
//         mai apropiat nod in embedding (legatura semantica, nu fizica).
//   - ANCORE CONTEXTUALE: nod = (token, domeniu) => "for"@lang != "for"@code.
//   - TOKENI VIRTUALI: stari/intentii ([cod_valid], ...) traversate de planner
//     (planner-ul devine "compilator": planuieste structura inainte de cuvinte).
//   - AUTO-REWRITE: o muchie care duce intr-o fundatura poate fi PENALIZATA
//     (confidence redus) => "uitare"/auto-corectie.
//   - Memoria lumii = GRAF, nu greutatile. Traversare determinista =>
//     anti-halucinatie: daca nu exista drum (si nu e mod creativ) => "Nu stiu".
//   - Transformerul e DOAR encoder de embeddings / context, nu baza de date.
//
//  C++17, un singur fisier, fara librarii externe.
//  Compilare:  g++ -O2 -std=c++17 concept_engine.cpp -o ce
//  Rulare:     ./ce [knowledge.txt]      (sau:  ./ce < knowledge.txt)
// ============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
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
    for(char c:s){ if(c==' '||c=='\t'){ if(!w.empty()) r.push_back(w); w.clear(); } else w+=c; }
    if(!w.empty()) r.push_back(w);
    return r;
}
string trim(const string& s){
    size_t a=s.find_first_not_of(" \t\r\n"); if(a==string::npos) return "";
    size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1);
}
float dot(const vector<float>&a,const vector<float>&b){
    float s=0; for(size_t i=0;i<a.size();i++) s+=a[i]*b[i]; return s;
}
float norm(const vector<float>&a){ return sqrt(dot(a,a)+1e-9f); }
float cosSim(const vector<float>&a,const vector<float>&b){
    return dot(a,b)/(norm(a)*norm(b));
}

// ============================================================================
//  PARTEA A — ENCODER NEURAL (Transformer mic): embeddings + vector de context
// ----------------------------------------------------------------------------
//  Invata next-token pe corpus; ca efect secundar embeddingurile tokenilor din
//  contexte similare devin apropiate. In generare il folosim ca SCORER: scoate
//  un vector de context din secventa de pana acum, cu care sortam vecinii din
//  graf. NU stocheaza fapte.
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
    vector<float> tokenVec(int id){vector<float> v(D);for(int i=0;i<D;i++)v[i]=tokEmb.at(id,i);return v;}
    // vector de context = hidden-ul ultimei pozitii din secventa (folosit la generare)
    vector<float> contextVec(const vector<int>& ids){
        if(ids.empty()) return vector<float>(D,0);
        Cache c=forward(ids); return c.hid.back();
    }
};

// ============================================================================
//  PARTEA B — TOKENIZER cu ANCORE CONTEXTUALE (token, domeniu)
// ----------------------------------------------------------------------------
//  Un nod nu e doar un cuvant, ci un cuvant intr-un DOMENIU. Stocam intern
//  "domeniu|cuvant" => acelasi "for" in NL si in cod sunt noduri diferite.
//  Tokenii virtuali (suprafata incepe cu '[') sunt stari/intentii structurale.
// ============================================================================
struct Tokenizer {
    unordered_map<string,int> id; vector<string> word;
    static string key(const string& surface,const string& domain){
        return domain.empty()? surface : domain+"|"+surface;
    }
    int add(const string&s){auto it=id.find(s);if(it!=id.end())return it->second;
        int x=word.size();id[s]=x;word.push_back(s);return x;}
    int addDom(const string& surface,const string& domain){ return add(key(surface,domain)); }
    int get(const string&s)const{auto it=id.find(s);return it==id.end()?-1:it->second;}
    int getDom(const string& surface,const string& domain)const{ return get(key(surface,domain)); }
    string name(int id)const{return (id>=0&&id<(int)word.size())?word[id]:"?";}
    // afisaj prietenos: "domeniu|cuvant" -> "cuvant@domeniu"
    string pretty(int id)const{
        string s=name(id); auto p=s.find('|');
        return p==string::npos? s : s.substr(p+1)+"@"+s.substr(0,p);
    }
    string surface(int id)const{ string s=name(id); auto p=s.find('|'); return p==string::npos?s:s.substr(p+1); }
    bool isVirtual(int id)const{ return !surface(id).empty() && surface(id)[0]=='['; }
};

// ============================================================================
//  PARTEA C — DESCOPERIRE DE ROLURI (fara adnotare manuala)
// ----------------------------------------------------------------------------
//  Care token e RELATIE (conector) se afla distributional: apare PREDOMINANT
//  in pozitie de mijloc. Sursa/tinta/corp ies din pozitie + aritate.
// ============================================================================
struct RoleDiscovery {
    unordered_set<int> relationTokens;
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
            if(m>0 && m*2>total) relationTokens.insert(t);   // majoritate stricta in mijloc
        }
    }
    bool isRelation(int t)const{ return relationTokens.count(t)>0; }
};

struct RelationVocab {
    unordered_map<string,int> id; vector<string> pattern;
    int discover(const string&p){auto it=id.find(p);if(it!=id.end())return it->second;
        int x=pattern.size();id[p]=x;pattern.push_back(p);return x;}
    int get(const string&p)const{auto it=id.find(p);return it==id.end()?-1:it->second;}
};

struct Parsed {
    bool ok=false, isDefinition=false;
    int src=-1, dst=-1, relation_id=-1;
    vector<int> body; string relPattern;
};

// src = capatul de dinaintea primei relatii; rel = run-ul de conectori;
// dupa rel: 1 token => FAPT (dst); >=2 tokeni => DEFINITIE (body).
Parsed parseByDiscoveredRoles(const vector<int>& ids, const RoleDiscovery& rd,
                              Tokenizer& tok, RelationVocab& rv){
    Parsed p;
    if(ids.size()<3) return p;
    int r0=-1;
    for(size_t i=1;i<ids.size();i++){ if(rd.isRelation(ids[i])){ r0=i; break; } }
    if(r0<=0) return p;
    p.src=ids[r0-1];
    int r1=r0; string relPat;
    while(r1<(int)ids.size() && rd.isRelation(ids[r1])){
        if(!relPat.empty()) relPat+=" ";
        relPat+=tok.surface(ids[r1]); r1++;
    }
    int rem=(int)ids.size()-r1;
    if(rem<=0) return p;
    p.relPattern=relPat; p.relation_id=rv.discover(relPat);
    if(rem==1){ p.dst=ids[r1]; p.ok=true; }
    else { p.isDefinition=true; for(int i=r1;i<(int)ids.size();i++) p.body.push_back(ids[i]); p.ok=true; }
    return p;
}

// ============================================================================
//  PARTEA D — GRAPH MEMORY (noduri + muchii). Suporta penalizare (auto-rewrite).
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
    // AUTO-REWRITE: scade confidence-ul unei muchii (uitare / penalizare)
    bool penalize(int src,int dst,int rel,float factor){
        auto it=outAdj.find(src); if(it==outAdj.end())return false;
        for(int ei:it->second) if(edges[ei].dst==dst&&edges[ei].relation_id==rel){
            edges[ei].confidence*=factor; return true; }
        return false;
    }
};

// ============================================================================
//  PARTEA E — CONCEPT MEMORY (abstractie emergenta = tinta is-a cu >=N surse)
// ============================================================================
struct ConceptMemory {
    set<int> concepts;
    unordered_map<int,set<int>> instances;
    unordered_map<int,int> instanceOf;
    void discover(const GraphMemory& g,int minInst,const set<int>& absRels){
        map<pair<int,int>,set<int>> bucket;
        for(auto& e:g.edges) if(!e.derived && absRels.count(e.relation_id))
            bucket[{e.dst,e.relation_id}].insert(e.src);
        for(auto& kv:bucket){
            if((int)kv.second.size()>=minInst){
                int concept=kv.first.first; concepts.insert(concept);
                for(int inst:kv.second){ instances[concept].insert(inst); instanceOf[inst]=concept; }
            }
        }
    }
    bool isConcept(int id)const{return concepts.count(id);}
};

// ============================================================================
//  PARTEA F — DEFINITION MEMORY (definitii descoperite prin aritate)
// ============================================================================
struct DefinitionMemory {
    unordered_map<int,vector<int>> def;
    unordered_map<int,int> defRelation;
    void add(int term,int rel,const vector<int>& body){ def[term]=body; defRelation[term]=rel; }
    bool has(int term)const{return def.count(term);}
};

// ============================================================================
//  PARTEA G — RELATION MODES (abstractie / lant / scop, descoperite din graf)
// ============================================================================
enum RelMode { PLAIN=0, ABSTRACTION=1, CHAIN=2, GOAL=3 };
const char* modeName(int m){
    switch(m){case ABSTRACTION:return "ABSTRACTION(is-a)";case CHAIN:return "CHAIN(cauza->efect)";
              case GOAL:return "GOAL(scop)";default:return "PLAIN";}
}
struct RelationModes {
    unordered_map<int,int> mode;
    set<int> goalTargets;
    void discover(const GraphMemory& g, const RelationVocab& rv){
        map<int,map<int,set<int>>> dstSrc; map<int,set<int>> dsts;
        for(auto& e:g.edges){ if(e.derived) continue;
            dstSrc[e.relation_id][e.dst].insert(e.src); dsts[e.relation_id].insert(e.dst); }
        for(auto& kv:dstSrc){
            int rel=kv.first;
            int maxFanIn=0, classes=0;
            for(auto& d:kv.second){ int f=d.second.size(); maxFanIn=max(maxFanIn,f); if(f>=2) classes++; }
            int cont=0,tot=0; for(int d:dsts[rel]){ tot++; if(g.hasOutgoing(d)) cont++; }
            float chainFrac = tot? (float)cont/tot : 0.f;
            bool multiWord = split(rv.pattern[rel]).size()>=2;
            if(multiWord){ mode[rel]=GOAL; for(int d:dsts[rel]) goalTargets.insert(d); }
            else if(classes>=2) mode[rel]=ABSTRACTION;
            else if(chainFrac>=0.5f) mode[rel]=CHAIN;
            else if(maxFanIn>=2) mode[rel]=ABSTRACTION;
            else mode[rel]=PLAIN;
        }
    }
    int of(int rel)const{ auto it=mode.find(rel); return it==mode.end()?PLAIN:it->second; }
    bool isAbstraction(int rel)const{ return of(rel)==ABSTRACTION; }
    bool isGoalTarget(int node)const{ return goalTargets.count(node)>0; }
};

// ============================================================================
//  PARTEA H — RULE ENGINE (mostenire de categorie + forward-chaining)
// ============================================================================
struct DiscoveredRule { string desc; int applications=0; };
struct RuleEngine {
    vector<DiscoveredRule> run(GraphMemory& g, const RelationModes& rm, const RelationVocab& rv){
        vector<DiscoveredRule> rules;
        set<int> absRels; for(auto& kv:rm.mode) if(kv.second==ABSTRACTION) absRels.insert(kv.first);
        if(absRels.empty()) return rules;
        for(int ar:absRels){
            DiscoveredRule r; r.desc = "daca X --(orice)--> Y si Y --[" + rv.pattern[ar] +
                "]--> Z atunci X --[" + rv.pattern[ar] + "]--> Z  (mostenire de categorie)";
            rules.push_back(r);
        }
        bool changed=true; int guard=0;
        while(changed && guard++<8){
            changed=false; int m=g.edges.size();
            for(int i=0;i<m;i++){
                Edge e=g.edges[i];
                for(int ar:absRels){
                    for(auto* ae:g.query(e.dst,ar)){
                        if(e.src==ae->dst) continue;
                        float conf=e.confidence*ae->confidence*0.9f;
                        size_t before=g.edges.size();
                        g.addEdge(e.src,ae->dst,ar,conf,true);
                        if(g.edges.size()>before){ changed=true;
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
//  PARTEA I — REASONING ENGINE (traversare multi-pas + scop + planner)
// ============================================================================
struct Chain { vector<int> nodes; vector<int> rels; float confidence=0; };

struct ReasoningEngine {
    GraphMemory& g; ConceptMemory& cm; DefinitionMemory& dm; RelationModes& rm;
    Tokenizer& tok; RelationVocab& rv;
    ReasoningEngine(GraphMemory&G,ConceptMemory&C,DefinitionMemory&D,RelationModes&M,
                    Tokenizer&T,RelationVocab&R):g(G),cm(C),dm(D),rm(M),tok(T),rv(R){}

    vector<int> direct(int src,int rel){ vector<int> r; for(auto*e:g.query(src,rel)) r.push_back(e->dst); return r; }
    string nodeName(int id){ return tok.pretty(id); }

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
                ns.rels.push_back(e->relation_id); ns.conf*=e->confidence; q.push(ns);
            }
        }
        return Chain{};
    }
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
    string why(int actor){
        int goal=-1;
        struct St{int node;vector<int> nodes;vector<int> rels;float conf;};
        queue<St> q; q.push({actor,{actor},{},1.0f});
        set<int> vis; vis.insert(actor);
        while(!q.empty()){
            St s=q.front();q.pop();
            if(s.node!=actor && rm.isGoalTarget(s.node)){ goal=s.node; break; }
            if((int)s.nodes.size()>=8) continue;
            for(auto*e:g.outgoing(s.node)){ if(vis.count(e->dst))continue; vis.insert(e->dst);
                St ns=s; ns.node=e->dst; ns.nodes.push_back(e->dst);
                ns.rels.push_back(e->relation_id); ns.conf*=e->confidence; q.push(ns); }
        }
        if(goal<0){ Chain ch=bestChain(actor,8); if(ch.nodes.size()<2) return "Nu stiu.";
                    return "Pentru ca: " + verbalize(ch); }
        string out = "Pentru a obtine " + nodeName(goal);
        Chain cont = bestChain(goal,6);
        if(cont.nodes.size()>=2){
            string just;
            for(size_t i=0;i+1<cont.nodes.size();i++){ if(!just.empty()) just+="; ";
                just += nodeName(cont.nodes[i])+" "+rv.pattern[cont.rels[i]]+" "+nodeName(cont.nodes[i+1]); }
            out += " (necesar pentru: " + just + ")";
        }
        return out + ".";
    }
    string expandDefinition(int term){
        if(!dm.has(term)) return "";
        auto& body=dm.def.at(term);
        string s=nodeName(term)+" := ";
        for(size_t i=0;i<body.size();i++) s+=(i?" ":"")+nodeName(body[i]);
        return s;
    }
};

// ============================================================================
//  PARTEA J — GENERATOR: "NEXT-TOKEN PE GRAF"
// ----------------------------------------------------------------------------
//  Bucla de generare ("The Loop"): din nodul curent, candidatii sunt VECINII
//  fizici din graf. Transformerul scoate vectorul de context al secventei de
//  pana acum; fiecare candidat e scorat = confidence * (0.5 + 0.5*cosSim).
//    - mod determinist: doar muchii reale (anti-halucinatie, oprire la fundatura)
//    - mod creativ: la fundatura, "sare" la cel mai apropiat nod in embedding
//      (legatura semantica/metafora, fara muchie fizica).
// ============================================================================
struct GenStep { int node; int rel; bool jump; };   // rel==-1 & jump => salt creativ
struct Generated { vector<GenStep> steps; };

struct Generator {
    GraphMemory& g; Encoder& enc; Tokenizer& tok; RelationModes& rm;
    Generator(GraphMemory&G,Encoder&E,Tokenizer&T,RelationModes&M):g(G),enc(E),tok(T),rm(M){}

    Generated generate(int seed,int maxLen,bool creative){
        Generated out; out.steps.push_back({seed,-1,false});
        vector<int> ids={seed}; set<int> visited={seed}; int cur=seed;
        for(int step=0; step<maxLen; step++){
            vector<const Edge*> cand;
            for(auto*e:g.outgoing(cur)) if(!visited.count(e->dst)) cand.push_back(e);
            const Edge* pick=nullptr; int jump=-1;
            if(!cand.empty()){
                auto cv=enc.contextVec(ids); float bs=-1e9f;
                for(auto*e:cand){
                    float sim=cosSim(cv,enc.tokenVec(e->dst));
                    float sc=e->confidence*(0.5f+0.5f*sim);
                    if(sc>bs){bs=sc;pick=e;}
                }
            } else if(creative){
                auto cv=enc.contextVec(ids); float bs=0.20f;   // prag minim de "rezonanta"
                for(int t=0;t<(int)tok.word.size();t++){
                    if(visited.count(t)) continue;
                    float sim=cosSim(cv,enc.tokenVec(t));
                    if(sim>bs){bs=sim;jump=t;}
                }
            }
            if(pick){ ids.push_back(pick->dst); visited.insert(pick->dst);
                      out.steps.push_back({pick->dst,pick->relation_id,false}); cur=pick->dst; }
            else if(jump>=0){ ids.push_back(jump); visited.insert(jump);
                      out.steps.push_back({jump,-1,true}); cur=jump; }
            else break;   // anti-halucinatie: nu inventa muchii
        }
        return out;
    }
    string show(const Generated& gen, const RelationVocab& rv){
        string s=tok.pretty(gen.steps[0].node);
        for(size_t i=1;i<gen.steps.size();i++){
            const auto& st=gen.steps[i];
            s += st.jump ? string("  ~(salt)~> ") : "  -("+rv.pattern[st.rel]+")-> ";
            s += tok.pretty(st.node);
        }
        return s;
    }
};

// ============================================================================
//  PARTEA K — LOADER: invata PRIN CITIRE dintr-un corpus extern (nu din cod)
// ----------------------------------------------------------------------------
//  Sectiuni [FACTS] / [DEFINITIONS], antet optional @domeniu (ancora).
//  '#' = comentariu. Liniile sunt singura sursa de cunostinte.
// ============================================================================
struct Corpus {
    vector<vector<int>> factIds, defIds, all;   // tokenizate (cu domeniu aplicat)
};
Corpus loadCorpus(istream& in, Tokenizer& tok){
    Corpus c; string line; string section="FACTS", domain="";
    while(getline(in,line)){
        line=trim(line);
        if(line.empty()||line[0]=='#') continue;
        if(line[0]=='['){
            string inside=line.substr(1, line.find(']')==string::npos? line.size()-1 : line.find(']')-1);
            auto parts=split(inside); domain="";
            if(!parts.empty()){
                section = parts[0];
                for(size_t i=1;i<parts.size();i++) if(parts[i][0]=='@') domain=parts[i].substr(1);
            }
            continue;
        }
        auto words=split(line); if(words.empty()) continue;
        vector<int> ids; for(auto& w:words) ids.push_back(tok.addDom(w,domain));
        c.all.push_back(ids);
        if(section=="DEFINITIONS") c.defIds.push_back(ids);
        else c.factIds.push_back(ids);
    }
    return c;
}

// ============================================================================
//  PARTEA L — MAIN
// ============================================================================
int main(int argc,char** argv){
    Tokenizer tok; RelationVocab rv; GraphMemory graph;
    ConceptMemory concepts; DefinitionMemory defs; RelationModes modes;

    // ---- INVATA PRIN CITIRE: corpus din argv[1] / "knowledge.txt" / stdin ----
    Corpus corp;
    string path = argc>1? argv[1] : "knowledge.txt";
    { ifstream f(path);
      if(f){ cout<<"== Citesc cunostinte din \""<<path<<"\" ==\n"; corp=loadCorpus(f,tok); }
      else { cout<<"== Citesc cunostinte din stdin ==\n"; corp=loadCorpus(cin,tok); } }
    if(corp.all.empty()){ cout<<"(corpus gol — nimic de invatat)\n"; return 0; }

    // ---- antreneaza encoderul (DOAR pentru embeddings / vector de context) ----
    Encoder enc(tok.word.size(), 16, 32, 64, 4);
    cout<<"== Antrenez encoderul (embeddings/context) ==\n";
    float lr=0.005f,wd=0.001f,step=0;
    for(int ep=0;ep<150;ep++){
        float tot=0;int cc=0;
        vector<int> ord(corp.all.size()); for(int i=0;i<(int)ord.size();i++)ord[i]=i;
        shuffle(ord.begin(),ord.end(),rng);
        for(int idx:ord){enc.zerograd();tot+=enc.trainSeq(corp.all[idx]);step++;enc.step(lr,step,wd);cc++;}
        if(ep%50==0)cout<<"  epoch "<<ep<<"  loss "<<tot/cc<<"\n";
    }
    cout<<"Done.\n\n";

    // ---- DESCOPERA rolurile din fapte; parseaza -> graf ----
    RoleDiscovery roles; roles.learn(corp.factIds);
    for(auto& ids:corp.factIds){
        Parsed p=parseByDiscoveredRoles(ids,roles,tok,rv);
        if(p.ok && !p.isDefinition) graph.addEdge(p.src,p.dst,p.relation_id,0.95f);
    }
    // ---- definitii: conectorul = tokenul recurent pe pozitia 1 ----
    {
        unordered_map<int,int> at1;
        for(auto& ids:corp.defIds) if(ids.size()>=3) at1[ids[1]]++;
        int connector=-1,best=0; for(auto&kv:at1) if(kv.second>best){best=kv.second;connector=kv.first;}
        for(auto& ids:corp.defIds){
            if((int)ids.size()<3 || ids[1]!=connector) continue;
            int rel=rv.discover(tok.surface(connector));
            vector<int> body(ids.begin()+2,ids.end());
            defs.add(ids[0],rel,body);
        }
    }

    // ---- moduri, concepte, reguli ----
    modes.discover(graph, rv);
    set<int> absRels; for(auto&kv:modes.mode) if(kv.second==ABSTRACTION) absRels.insert(kv.first);
    concepts.discover(graph, 2, absRels);
    RuleEngine ruleEng; auto rules = ruleEng.run(graph, modes, rv);

    ReasoningEngine reason(graph,concepts,defs,modes,tok,rv);
    Generator gen(graph,enc,tok,modes);

    // ---- ALEGE automat entitati de demonstrat (nimic hardcodat) ----
    int demoActor=-1, demoGoal=-1;
    for(int cId:concepts.concepts){
        for(int inst:concepts.instances[cId]){
            for(int goal:modes.goalTargets){
                if(reason.plan(inst,goal,8).nodes.size()>=2){ demoActor=inst; demoGoal=goal; break; }
            }
            if(demoActor>=0) break;
        }
        if(demoActor>=0) break;
    }
    int demoConcept = concepts.concepts.empty()? -1 : *concepts.concepts.rbegin();
    // un nod de cod (domeniu "code") si un token virtual, pentru planner-compilator
    int codeSeed=-1, virtGoal=-1;
    for(int i=0;i<(int)tok.word.size();i++){
        string nm=tok.name(i);
        if(codeSeed<0 && nm.rfind("code|",0)==0 && graph.hasOutgoing(i) && !tok.isVirtual(i)) codeSeed=i;
        if(virtGoal<0 && tok.isVirtual(i)) virtGoal=i;
    }

    // ================== OUTPUT ==================
    cout<<"== 1. ROLURI DESCOPERITE (tokeni-relatie) ==\n  ";
    { vector<string> rs; for(int t:roles.relationTokens) rs.push_back(tok.pretty(t));
      sort(rs.begin(),rs.end()); bool f=true;
      for(auto&s:rs){cout<<(f?"":", ")<<s;f=false;} cout<<"\n"; }

    cout<<"\n== 2. RELATII + MOD emergent ==\n";
    set<int> defRels; for(auto&kv:defs.defRelation) defRels.insert(kv.second);
    for(int i=0;i<(int)rv.pattern.size();i++){
        string m = defRels.count(i)? "DEFINITION(definitie)" : modeName(modes.of(i));
        cout<<"  relation_id "<<i<<"  <- \""<<rv.pattern[i]<<"\"   mod: "<<m<<"\n";
    }

    cout<<"\n== 3. CONCEPTE descoperite (abstractie) ==\n";
    for(int cId:concepts.concepts){
        cout<<"  concept: "<<tok.pretty(cId)<<"   instante: ";
        bool first=true; for(int inst:concepts.instances[cId]){cout<<(first?"":", ")<<tok.pretty(inst);first=false;}
        cout<<"\n";
    }

    cout<<"\n== 4. DEFINITII descoperite (prin aritate) ==\n";
    for(auto&kv:defs.def){
        cout<<"  "<<tok.pretty(kv.first)<<" := ";
        for(size_t i=0;i<kv.second.size();i++)cout<<(i?" ":"")<<tok.pretty(kv.second[i]);
        cout<<"\n";
    }

    cout<<"\n== 5. REGULI + forward-chaining ==\n";
    for(auto& r:rules) cout<<"  - "<<r.desc<<"   [aplicata de "<<r.applications<<" ori]\n";

    cout<<"\n== 6. ANCORE CONTEXTUALE (acelasi cuvant, domenii izolate) ==\n";
    {
        // arata fiecare suprafata care exista in >=2 domenii => noduri distincte
        map<string,vector<int>> bySurface;
        for(int i=0;i<(int)tok.word.size();i++) bySurface[tok.surface(i)].push_back(i);
        bool any=false;
        for(auto& kv:bySurface) if(kv.second.size()>=2){ any=true;
            cout<<"  \""<<kv.first<<"\" -> noduri distincte: ";
            bool f=true; for(int id:kv.second){cout<<(f?"":", ")<<"node#"<<id<<"("<<tok.pretty(id)<<")";f=false;}
            cout<<"\n";
        }
        if(!any) cout<<"  (niciun cuvant nu apare in mai multe domenii in acest corpus)\n";
    }

    cout<<"\n== 7. NEXT-TOKEN PE GRAF — generare DETERMINISTA (logica) ==\n";
    if(demoActor>=0){
        auto G=gen.generate(demoActor,8,false);
        cout<<"  seed auto = '"<<tok.pretty(demoActor)<<"'\n  "<<gen.show(G,rv)<<"\n";
    } else cout<<"  (niciun actor potrivit)\n";
    if(codeSeed>=0){
        auto G=gen.generate(codeSeed,8,false);
        cout<<"  seed cod  = '"<<tok.pretty(codeSeed)<<"'\n  "<<gen.show(G,rv)<<"\n";
    }

    cout<<"\n== 8. NEXT-TOKEN PE GRAF — generare CREATIVA (salt prin embedding) ==\n";
    if(demoConcept>=0){
        auto G=gen.generate(demoConcept,8,true);
        cout<<"  seed auto = '"<<tok.pretty(demoConcept)<<"'\n  "<<gen.show(G,rv)<<"\n";
        cout<<"  (\"~(salt)~>\" = legatura semantica fara muchie fizica)\n";
    }

    cout<<"\n== 9. AUTO-REWRITE (penalizare muchie -> alta generare) ==\n";
    if(demoActor>=0){
        auto G1=gen.generate(demoActor,6,false);
        cout<<"  inainte: "<<gen.show(G1,rv)<<"\n";
        if(G1.steps.size()>=2){
            int a=G1.steps[0].node, b=G1.steps[1].node, r=G1.steps[1].rel;
            graph.penalize(a,b,r,0.05f);   // muchia aleasa devine "nesigura"
            cout<<"  penalizez muchia "<<tok.pretty(a)<<" -("<<rv.pattern[r]<<")-> "<<tok.pretty(b)<<" (x0.05)\n";
            auto G2=gen.generate(demoActor,6,false);
            cout<<"  dupa:    "<<gen.show(G2,rv)<<"\n";
        }
    }

    cout<<"\n== 10. PLANNER-COMPILATOR (drum spre un token virtual) ==\n";
    if(virtGoal>=0){
        // alege automat un nod de start care POATE atinge starea virtuala [cod_valid]
        int planSeed=-1; Chain best;
        for(int i=0;i<(int)tok.word.size();i++){
            if(i==virtGoal || tok.isVirtual(i) || !graph.hasOutgoing(i)) continue;
            Chain p=reason.plan(i,virtGoal,8);
            if(p.nodes.size()>=2 && (planSeed<0 || p.nodes.size()>best.nodes.size())){ planSeed=i; best=p; }
        }
        if(planSeed>=0) cout<<"  plan "<<tok.pretty(planSeed)<<" -> "<<tok.pretty(virtGoal)<<": "<<reason.verbalize(best)<<"\n";
        else cout<<"  niciun nod nu atinge "<<tok.pretty(virtGoal)<<" (Nu stiu)\n";
    } else cout<<"  (niciun token virtual in corpus)\n";

    cout<<"\n== 11. RATIONAMENT \"DE CE?\" (cauza -> scop) ==\n";
    if(demoActor>=0)
        cout<<"  De ce actioneaza "<<tok.pretty(demoActor)<<"?\n    A: "<<reason.why(demoActor)<<"\n";

    cout<<"\n== 12. EXPANDARE prin DEFINITIE ==\n";
    for(auto&kv:defs.def){ cout<<"  "<<reason.expandDefinition(kv.first)<<"\n"; }

    cout<<"\n== 13. ANTI-HALUCINATIE ==\n";
    int dragon=tok.get("dragon"); // token inexistent in corpus
    cout<<"  Q: generare din 'dragon' (necunoscut)?\n  A: ";
    if(dragon<0) cout<<"Nu stiu (token absent din graf).\n";
    else { auto G=gen.generate(dragon,5,false); cout<<gen.show(G,rv)<<"\n"; }
    if(demoActor>=0){
        // o relatie inventata, care nu exista in vocabular => Nu stiu
        cout<<"  Q: raspuns pe o relatie inexistenta ('zboara')?\n  A: ";
        int r=rv.get("zboara");
        if(r<0) cout<<"Nu stiu (relatie neobservata).\n";
        else { auto outs=reason.direct(demoActor,r); cout<<(outs.empty()?"Nu stiu.":"...")<<"\n"; }
    }
    return 0;
}
