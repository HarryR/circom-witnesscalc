use std::{env, fs};
use std::collections::HashMap;
use num_bigint::BigUint;
use num_traits::{Num, ToBytes};
use circom_witnesscalc::{ast, vm2};
use circom_witnesscalc::ast::{Literal, Statement, AST};
use circom_witnesscalc::field::{FieldOps, U254};
use circom_witnesscalc::parser::parse_ast;
use circom_witnesscalc::vm2::OpCode;

struct Args {
    cvm_file: String,
    output_file: String,
}


fn parse_args() -> Args {
    let mut cvm_file: Option<String> = None;
    let mut output_file: Option<String> = None;

    let args: Vec<String> = env::args().collect();

    let usage = |err_msg: &str| {
        if !err_msg.is_empty() {
            eprintln!("ERROR:");
            eprintln!("    {}", err_msg);
            eprintln!();
        }
        eprintln!("USAGE:");
        eprintln!("    {} <cvm_file> <output_path> [OPTIONS]", args[0]);
        eprintln!();
        eprintln!("ARGUMENTS:");
        eprintln!("    <cvm_file>    Path to the CVM file with compiled circuit");
        eprintln!("    <output_path> File where the witness will be saved");
        eprintln!();
        eprintln!("OPTIONS:");
        eprintln!("    -h | --help                Display this help message");
        let exit_code = if !err_msg.is_empty() { 1i32 } else { 0i32 };
        std::process::exit(exit_code);
    };

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            usage("");
        } else if args[i].starts_with("-") {
            usage(format!("Unknown option: {}", args[i]).as_str());
        } else if cvm_file.is_none() {
            cvm_file = Some(args[i].clone());
        } else if output_file.is_none() {
            output_file = Some(args[i].clone());
        }
        i += 1;
    }
    Args {
        cvm_file: cvm_file.unwrap_or_else(|| { usage("missing CVM file"); String::new() }),
        output_file: output_file.unwrap_or_else(|| { usage("missing output file"); String::new() }),
    }
}

fn main() {
    let args = parse_args();

    let program_text = fs::read_to_string(&args.cvm_file).unwrap();
    let program = parse_ast(&mut(program_text.as_str())).unwrap();
    let bn254 = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
    match &program.prime {
        bn254 => {
            compile::<U254>(&program);
        }
    }
    println!("OK {}", args.cvm_file);
}

fn compile<T>(tree: &ast::AST)
where
    T: FieldOps {

    for t in tree.templates.iter() {
        let compiled_template = compile_template::<T>(t);
        println!("Template: {}", t.name);
        println!("Compiled code len: {}", compiled_template.code.len());
    }
}

struct TemplateCompilationContext {
    code: Vec<u8>,
    ff_variable_indexes: HashMap<String, i64>,
    i64_variable_indexes: HashMap<String, i64>,
}

impl TemplateCompilationContext {
    fn new() -> Self {
        Self {
            code: vec![],
            ff_variable_indexes: HashMap::new(),
            i64_variable_indexes: HashMap::new(),
        }
    }

    fn get_ff_variable_index(&mut self, var_name: &str) -> i64 {
        let next_idx = self.ff_variable_indexes.len() as i64;
        *self.ff_variable_indexes
            .entry(var_name.to_string()).or_insert(next_idx)
    }
    fn get_i64_variable_index(&mut self, var_name: &str) -> i64 {
        let next_idx = self.i64_variable_indexes.len() as i64;
        *self.i64_variable_indexes
            .entry(var_name.to_string()).or_insert(next_idx)
    }
}

fn operand_ff<T>(ctx: &mut TemplateCompilationContext, operand: &ast::FfOperand)
where
    T: FieldOps {

    match operand {
        ast::FfOperand::Literal(v) => {
            ctx.code.push(OpCode::PushI64 as u8);
            let x = T::from_le_bytes(v.to_le_bytes().as_slice()).unwrap();
            ctx.code.extend_from_slice(x.to_le_bytes().as_slice());
        }
        ast::FfOperand::Variable(var_name) => {
            let var_idx = ctx.get_ff_variable_index(var_name);
            ctx.code.push(OpCode::LoadVariableFf as u8);
            ctx.code.extend_from_slice(var_idx.to_le_bytes().as_slice());
        }
    }
}

fn operand_i64(
    ctx: &mut TemplateCompilationContext, operand: &ast::I64Operand) {

    match operand {
        ast::I64Operand::Literal(v) => {
            ctx.code.push(OpCode::PushI64 as u8);
            ctx.code.extend_from_slice(v.to_le_bytes().as_slice());
        }
        ast::I64Operand::Variable(var_name) => {
            let var_idx = ctx.get_i64_variable_index(var_name);
            ctx.code.push(OpCode::LoadVariableI64 as u8);
            ctx.code.extend_from_slice(var_idx.to_le_bytes().as_slice());
        }
    }
}

fn expression<T>(ctx: &mut TemplateCompilationContext, expr: &ast::Expr)
where
    T: FieldOps {

    match expr {
        ast::Expr::GetSignal(operand) => {
            operand_i64(ctx, operand);
            ctx.code.push(OpCode::LoadSignal as u8);
        }
        ast::Expr::FfMul(lhs, rhs) => {
            operand_ff::<T>(ctx, rhs);
            operand_ff::<T>(ctx, lhs);
            ctx.code.push(OpCode::OpMul as u8);
        },
        ast::Expr::FfAdd(lhs, rhs) => {
            operand_ff::<T>(ctx, rhs);
            operand_ff::<T>(ctx, lhs);
            ctx.code.push(OpCode::OpAdd as u8);
        },
    }
}

fn instruction<T>(
    ctx: &mut TemplateCompilationContext, inst: &ast::TemplateInstruction)
where
    T: FieldOps {

    match inst {
        ast::TemplateInstruction::Assignment(assignment) => {
            expression::<T>(ctx, &assignment.value);
            ctx.code.push(OpCode::StoreVariable as u8);
            let var_idx = ctx.get_ff_variable_index(&assignment.dest);
            ctx.code.extend_from_slice(var_idx.to_le_bytes().as_slice());
        }
        ast::TemplateInstruction::Statement(statement) => {
            match statement {
                Statement::SetSignal { idx, value } => {
                    operand_ff::<T>(ctx, value);
                    operand_i64(ctx, idx);
                    ctx.code.push(OpCode::StoreSignal as u8);
                }
            }
        }
    }
}

fn compile_template<T>(t: &ast::Template) -> vm2::Template
where
    T: FieldOps {

    let mut ctx = TemplateCompilationContext::new();
    for i in &t.body {
        instruction::<T>(&mut ctx, &i);
    }

    vm2::Template {
        name: t.name.clone(),
        code: ctx.code,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example() {
        assert!(true);
    }
}