pragma circom 2.0.0;

function fnc1(a, b) {
  if (a == 3) {
    return [0, 1];
  } else {
    return [b*3, 4];
  }
}

template Tmpl1(n) {
    signal input a;
    signal input b;
    signal output c;
    var d[2] = fnc1(n, b);
    c <== b * a + d[1];
}

component main = Tmpl1(2);

// ff.mcall $fnc1_0 i64.1 i64.2  ff.2 x_14