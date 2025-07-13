#include <iostream>
using namespace std;


//sum of n natural nos
void sum(int i, int s){
    if (i < 1) {
        cout << s << endl;
        return;
    }
    sum(i-1,s+i);
}

//using backtracking
void bSum(int i, int &sum){
   if (i < 1) return;
    bSum(i - 1, sum); // recursion till base
    sum += i;  
}

//functional recursion
int sum(int n){
    if(n == 0) return 0;
    return n + sum(n-1);
}

int main(){
    int n;
    cin>>n;
    cout<<sum(n);
}