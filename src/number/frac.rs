#![allow(warnings)]
use num_traits::{Float, Num, One, Zero};

use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub, SubAssign,
};

use std::cmp::{Ordering, PartialOrd};

use std::fmt::{Debug, Display};

use std::sync::Mutex;

use crate::traits::{CheckedOps, CloneFloat, CloneNum, CloneNumAssign, Pow, PowAssign};

use crate::{impl_for_ref, impl_for_ref_assign};

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct Frac<T: CloneNum> {
    pub nume: T,
    pub deno: T,
}

impl<T: CloneNum> Frac<T> {
    pub fn new(nume: T, deno: T) -> Self {
        if deno.is_zero() {
            panic!("Zero Divition Error");
        } else if nume.is_zero() {
            return Frac {
                nume,
                deno: T::one(),
            };
        }
        let gcd = Self::gcd(nume.clone(), deno.clone());
        Frac {
            nume: nume / gcd.clone(),
            deno: deno / gcd,
        }
    }

    fn gcd(x: T, y: T) -> T {
        if y == T::zero() {
            x
        } else {
            Frac::gcd(y.clone(), (x % y.clone() + y.clone()) % y)
        }
    }

    fn reduce(&self) -> Self {
        let gcd = Frac::gcd(self.nume(), self.deno());
        Frac::new(self.nume() / gcd.clone(), self.deno() / gcd.clone())
    }

    pub fn nume(&self) -> T {
        self.nume.clone()
    }

    pub fn deno(&self) -> T {
        self.deno.clone()
    }

    pub fn pow<U: CloneNum>(&self, y: U) -> Self {
        let two = U::one() + U::one();
        if y == U::zero() {
            Self::one()
        } else if y.clone() % two.clone() == U::one() {
            self.clone() * self.pow(y - U::one())
        } else {
            let a = self.pow(y / two);
            a.clone() * a.clone()
        }
    }

    pub fn rec(self) -> Self {
        Self::new(self.deno(), self.nume())
    }
}

impl<T: CloneNumAssign> Frac<T> {
    fn reduce_assign(&mut self) {
        if self.deno.is_zero() {
            panic!("Zero Divition Error");
        } else if self.nume.is_zero() {
            self.deno = T::one();
            return;
        }
        let gcd = Self::gcd(self.nume(), self.deno());
        self.nume /= gcd.clone();
        self.deno /= gcd.clone();
    }
}

impl<T: Neg<Output = T> + CloneNum> Neg for Frac<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Frac::new(-self.nume, self.deno)
    }
}

impl<T: CloneNum> Add for Frac<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Frac::new(
            self.nume() * rhs.deno() + self.deno() * rhs.nume(),
            self.deno() * rhs.deno(),
        )
    }
}

impl<T: CloneNumAssign> AddAssign for Frac<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.nume = self.nume() * rhs.deno() + self.deno() * rhs.nume();
        self.deno = self.deno() * rhs.deno();
        self.reduce_assign();
    }
}

impl<T: CloneNum> Sub for Frac<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Frac::new(
            self.nume() * rhs.deno() - self.deno() * rhs.nume(),
            self.deno() * rhs.deno(),
        )
    }
}

impl<T: CloneNumAssign> SubAssign for Frac<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.nume = self.nume() * rhs.deno() - self.deno() * rhs.nume();
        self.deno = self.deno() * rhs.deno();
        self.reduce_assign();
    }
}

impl<T: CloneNum> Mul for Frac<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Frac::new(self.nume() * rhs.nume(), self.deno() * rhs.deno())
    }
}

impl<T: CloneNumAssign> MulAssign for Frac<T> {
    fn mul_assign(&mut self, rhs: Self) {
        self.nume = self.nume() * rhs.nume();
        self.deno = self.deno() * rhs.deno();
        self.reduce_assign();
    }
}

impl<T: CloneNum> Div for Frac<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Frac::new(self.nume() * rhs.deno(), self.deno() * rhs.nume())
    }
}

impl<T: CloneNumAssign> DivAssign for Frac<T> {
    fn div_assign(&mut self, rhs: Self) {
        self.nume = self.nume() * rhs.deno();
        self.deno = self.deno() * rhs.nume();
        self.reduce_assign();
    }
}

impl<T: CloneNum> Rem for Frac<T> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Frac::new(
            self.nume() * rhs.deno() % self.deno() * rhs.nume(),
            self.deno() * rhs.deno(),
        )
    }
}

impl<T: CloneNumAssign> RemAssign for Frac<T> {
    fn rem_assign(&mut self, rhs: Self) {
        self.nume = self.nume() * rhs.deno() % self.deno() * rhs.nume();
        self.deno = self.deno() * rhs.deno();
        self.reduce_assign();
    }
}

impl<T: CloneNum, U: CloneNum> Pow<U> for Frac<T> {
    type Output = Self;

    fn pow(self, rhs: U) -> Self::Output {
        let two = U::one() + U::one();
        if rhs == U::zero() {
            Self::one()
        } else if rhs.clone() % two.clone() == U::one() {
            self.clone() * self.pow(rhs - U::one())
        } else {
            let a = self.pow(rhs / two);
            a.clone() * a.clone()
        }
    }
}

impl<T: CloneNumAssign, U: CloneNum> PowAssign<U> for Frac<T> {
    fn pow_assign(&mut self, rhs: U) {
        *self = self.clone().pow(rhs);
    }
}

impl<T: Not<Output = T> + CloneNum> Not for Frac<T> {
    type Output = Self;

    fn not(self) -> Self::Output {
        Frac::new(!self.nume, !self.deno)
    }
}

type U<T> = Frac<T>;

impl_for_ref!(U<T>, Add, add);

impl_for_ref!(U<T>, Sub, sub);

impl_for_ref!(U<T>, Mul, mul);

impl_for_ref!(U<T>, Div, div);

impl_for_ref!(U<T>, Rem, rem);

impl_for_ref_assign!(U<T>, AddAssign, add_assign);

impl_for_ref_assign!(U<T>, SubAssign, sub_assign);

impl_for_ref_assign!(U<T>, MulAssign, mul_assign);

impl_for_ref_assign!(U<T>, DivAssign, div_assign);

impl_for_ref_assign!(U<T>, RemAssign, rem_assign);

impl<T: CloneNum> Zero for Frac<T> {
    fn zero() -> Self {
        Frac::new(T::zero(), T::one())
    }

    fn is_zero(&self) -> bool {
        self.nume.is_zero()
    }

    fn set_zero(&mut self) {
        self.nume = T::zero();
        self.deno = T::one();
    }
}

impl<T: CloneNum> One for Frac<T> {
    fn one() -> Self {
        Frac::new(T::one(), T::one())
    }

    fn is_one(&self) -> bool {
        self.nume == self.deno
    }

    fn set_one(&mut self) {
        self.nume = T::one();
        self.deno = T::one();
    }
}

impl<T: CloneNum> PartialOrd for Frac<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let s = self.reduce();
        let r = other.reduce();
        if s.clone() == r.clone() {
            Some(Ordering::Equal)
        } else if s.clone() > r.clone() {
            Some(Ordering::Greater)
        } else if s.clone() < r.clone() {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

impl<T: CloneNum> From<T> for Frac<T> {
    fn from(value: T) -> Self {
        Frac::new(value, T::one())
    }
}

impl<T: CloneNum + Display> Display for Frac<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} / {}", self.nume, self.deno)
    }
}

impl<T: CloneNum> Num for Frac<T> {
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        match T::from_str_radix(str, radix) {
            Ok(x) => Ok(Frac::new(x, T::one())),
            Err(e) => Err(e),
        }
    }
}

impl<T: CloneNum + Default> Default for Frac<T> {
    fn default() -> Self {
        Frac::new(T::default(), T::one())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frac_in_frac() {
        let x = Frac::new(Frac::new(1, 2), Frac::new(2, 3));
        println!("{}", x);
    }

    #[test]
    fn float_frac() {
        let x = Frac::new(1.1, 3.7);
        println!("{}", x);
    }

    #[test]
    fn minus_frac() {
        let x = Frac::new(-3, 5);
        let y = Frac::new(3, -5);
        assert_eq!(x, y);
    }
}
