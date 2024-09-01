#![allow(warnings)]
use crate::traits::CloneNum;

fn comb<T: CloneNum + PartialOrd>(n: T, r: T) -> T {
    let mut x = T::one();
    let mut i = T::one();
    while i.clone() <= r.clone() {
        x = x * i.clone();
        i = i + T::one();
    }
    let mut res = T::one();
    let mut i = n.clone() - r + T::one();
    while i <= n {
        res = res * i.clone();
        i = i + T::one();
    }
    res / x
}

fn perm<T: CloneNum + PartialOrd>(n: T, r: T) -> T {
    let mut res = T::one();
    let mut i = n.clone() - r + T::one();
    while i <= n {
        res = res * i.clone();
        i = i + T::one();
    }
    res
}
