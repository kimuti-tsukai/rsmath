#![allow(wwarnings)]

#[macro_export]
macro_rules! impl_for_ref {
    ($struct: ty, $trait: ident, $method: ident) => {
        impl<'a, 'b, T: crate::traits::CloneNum> $trait<&'a $struct> for &'b $struct {
            type Output = $struct;

            fn $method(self, rhs: &'a $struct) -> Self::Output {
                self.clone().$method(rhs.clone())
            }
        }
    };
}

#[macro_export]
macro_rules! impl_for_ref_assign {
    ($struct: ty, $trait: ident,$method: ident) => {
        impl<'a, T: crate::traits::CloneNumAssign> $trait<&'a $struct> for $struct {
            fn $method(&mut self, rhs: &'a $struct) {
                *self += rhs.clone();
            }
        }
    };
}

#[macro_export]
macro_rules! for_c {
    (($var: ident = $val: expr ; $cons: expr ; $p: expr) $process: expr) => {
        {
            let mut $var = $val;
            while $cons {
                $process;
                $p;
            }
        }
    };
    ((let mut $var: ident = $val: expr ; $cons: expr ; $p: expr) $process: expr) => {
        {
            let mut $var = $val;
            while $cons {
                $process;
                $p;
            }
        }
    };
}
