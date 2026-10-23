// Experimental flux grouping: superlevel H0 persistence and marker-controlled MSF.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
static int32_t root(std::vector<int32_t>& p, int32_t i) {
  while (p[i]!=i) { p[i]=p[p[i]]; i=p[i]; } return i;
}
extern "C" int64_t persistence(int32_t n, int64_t m, const int32_t* edges,
 const int64_t* order, const double* s, double* death, int32_t* elder,
 int32_t* tree) {
 try {
 std::vector<int32_t> p(n); std::iota(p.begin(),p.end(),0);
 std::fill(death,death+n,-std::numeric_limits<double>::infinity());
 std::fill(elder,elder+n,-1); int64_t count=0;
 for (int64_t k=0;k<m;k++) {
  int64_t j=order[k]; int32_t a=edges[2*j],b=edges[2*j+1];
  int32_t x=root(p,a),y=root(p,b); if(x==y)continue;
  if(s[x]<s[y] || (s[x]==s[y] && x>y))std::swap(x,y);
  death[y]=std::min(s[a],s[b]); elder[y]=x; p[y]=x;
  tree[2*count]=a;tree[2*count+1]=b;count++;
 }
 return count;
 } catch (...) { return -1; }
}
extern "C" int watershed(int32_t n,int64_t m,const int32_t* tree,
 const double* persistence,double threshold,int32_t* labels) {
 try {
 std::vector<int32_t> p(n),size(n,1),marker(n,-1);std::iota(p.begin(),p.end(),0);
 for(int32_t i=0;i<n;i++) if(persistence[i]>threshold) marker[i]=i;
 for(int64_t j=0;j<m;j++) {
  int32_t x=root(p,tree[2*j]),y=root(p,tree[2*j+1]);
  if(x==y)continue;
  if(marker[x]>=0 && marker[y]>=0)continue;
  if(size[x]<size[y] || (size[x]==size[y] && x>y))std::swap(x,y);
  p[y]=x;size[x]+=size[y];if(marker[x]<0)marker[x]=marker[y];
 }
 for(int32_t i=0;i<n;i++)labels[i]=marker[root(p,i)];
 return 0;
 } catch (...) { return -1; }
}
