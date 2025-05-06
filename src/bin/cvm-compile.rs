
use circom_witnesscalc::vm::ComponentTmpl;
use circom_witnesscalc::ast::Literal;

fn main() {
    let _x = Literal::I64(123);
    println!("OK");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example() {
        assert!(true);
    }
}