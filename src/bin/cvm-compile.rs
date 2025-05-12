use std::{env, fs};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, Write};
use std::path::Path;
use num_bigint::BigUint;
use num_traits::{Num, ToBytes};
use wtns_file::FieldElement;
use circom_witnesscalc::{ast, vm2, wtns_from_witness2};
use circom_witnesscalc::ast::{Statement};
use circom_witnesscalc::field::{FieldOps, U254};
use circom_witnesscalc::parser::parse_ast;
use circom_witnesscalc::vm2::{disassemble_instruction, execute, Circuit, OpCode};

struct Args {
    cvm_file: String,
    output_file: String,
    wtns_file: Option<String>,
    sym_file: String,
}

#[derive(Debug, thiserror::Error)]
enum CompilationError {
    #[error("Main template ID is not found")]
    MainTemplateIDNotFound,
    #[error("witness signal index is out of bounds")]
    WitnessSignalIndexOutOfBounds,
    #[error("witness signal is not set")]
    WitnessSignalNotSet,
    #[error("incorrect SYM file format: `{0}`")]
    IncorrectSymFileFormat(String),
}

fn parse_args() -> Args {
    let mut cvm_file: Option<String> = None;
    let mut output_file: Option<String> = None;
    let mut wtns_file: Option<String> = None;
    let mut sym_file: Option<String> = None;

    let args: Vec<String> = env::args().collect();

    let usage = |err_msg: &str| {
        if !err_msg.is_empty() {
            eprintln!("ERROR:");
            eprintln!("    {}", err_msg);
            eprintln!();
        }
        eprintln!("USAGE:");
        eprintln!("    {} <cvm_file> <sym_file> <output_path> [OPTIONS]", args[0]);
        eprintln!();
        eprintln!("ARGUMENTS:");
        eprintln!("    <cvm_file>    Path to the CVM file with compiled circuit");
        eprintln!("    <sym_file>    Path to the SYM file with signals description");
        eprintln!("    <output_path> File where the witness will be saved");
        eprintln!();
        eprintln!("OPTIONS:");
        eprintln!("    -h | --help                Display this help message");
        eprintln!("    --wtns                      If file is provided, the witness will be calculated and saved in this file");
        let exit_code = if !err_msg.is_empty() { 1i32 } else { 0i32 };
        std::process::exit(exit_code);
    };

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            usage("");
        } else if args[i] == "--wtns" {
            i += 1;
            if i >= args.len() {
                usage("missing argument for --wtns");
            }
            if wtns_file.is_some() {
                usage("multiple witness files");
            }
            wtns_file = Some(args[i].clone());
        } else if args[i].starts_with("-") {
            usage(format!("Unknown option: {}", args[i]).as_str());
        } else if cvm_file.is_none() {
            cvm_file = Some(args[i].clone());
        } else if sym_file.is_none() {
            sym_file = Some(args[i].clone());
        } else if output_file.is_none() {
            output_file = Some(args[i].clone());
        }
        i += 1;
    }
    Args {
        cvm_file: cvm_file.unwrap_or_else(|| { usage("missing CVM file"); String::new() }),
        output_file: output_file.unwrap_or_else(|| { usage("missing output file"); String::new() }),
        wtns_file,
        sym_file: sym_file.unwrap_or_else(|| { usage("missing SYM file"); String::new() }),
    }
}

fn main() {
    let args = parse_args();

    let program_text = fs::read_to_string(&args.cvm_file).unwrap();
    let program = parse_ast(&mut(program_text.as_str())).unwrap();
    let bn254 = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
    if program.prime == bn254 {
        let circuit = compile::<U254>(&program, &args.sym_file).unwrap();
        disassemble::<U254>(&circuit.templates);
        if args.wtns_file.is_some() {
            calculate_witness(&circuit, args.wtns_file.unwrap()).unwrap();
        }
    } else {
        eprintln!("ERROR: Unsupported prime field");
        std::process::exit(1);
    }

    println!(
        "OK, output is supposed to be saved in {}, but it is not implemented yet.",
        args.output_file);
}

fn input_signals_info(
    sym_file: &str,
    main_template_id: usize) -> Result<HashMap<String, usize>, Box<dyn Error>> {

    let mut m: HashMap<String, usize> = HashMap::new();
    let file = File::open(Path::new(sym_file))?;
    let reader = std::io::BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        let values: Vec<&str> = line.split(',').collect();
        if values.len() != 4 {
            return Err(Box::new(CompilationError::IncorrectSymFileFormat(
                format!("line should consist of 4 values: {}", line))));
        }

        let node_id = values[2].parse::<usize>()
            .map_err(|e| Box::new(
                CompilationError::IncorrectSymFileFormat(
                    format!("node_id should be a number: {}", e))))?;
        if node_id != main_template_id {
            continue
        }

        let signal_idx = values[0].parse::<usize>()
            .map_err(|e| Box::new(
                CompilationError::IncorrectSymFileFormat(
                    format!("signal_idx should be a number: {}", e))))?;

        m.insert(values[3].to_string(), signal_idx);
    }
    Ok(m)
}

fn calculate_witness<T: FieldOps>(circuit: &Circuit<T>, wtns_file: String) -> Result<(), Box<dyn Error>> {
    let mut signals: Vec<Option<T>> = vec![
        Some(T::one()),
        None,
        Some(T::from_str("3").unwrap()),
        Some(T::from_str("4").unwrap())
    ];
    execute(&circuit.templates, &mut signals, circuit.main_template_id)?;
    let wtns_data = witness::<T>(&signals, &circuit.witness, circuit.prime)?;
    // for (i, s) in signals.iter().enumerate() {
    //     if let Some(s) = s {
    //         println!("Signal[{}]: {}", i, s);
    //     } else {
    //         println!("Signal[{}]: None", i);
    //     }
    // }

    let mut file = File::create(Path::new(&wtns_file))?;
    file.write_all(&wtns_data)?;
    file.flush()?;
    println!("Witness saved to {}", wtns_file);
    Ok(())
}

fn witness<T: FieldOps>(signals: &[Option<T>], witness_signals: &[usize], prime: T) -> Result<Vec<u8>, CompilationError> {
    let mut result = Vec::with_capacity(witness_signals.len());

    for &idx in witness_signals {
        if idx >= signals.len() {
            return Err(CompilationError::WitnessSignalIndexOutOfBounds)
        }

        match signals[idx] {
            Some(s) => result.push(s),
            None => return Err(CompilationError::WitnessSignalNotSet)
        }
    }

    match T::BYTES {
        8 => {
            let vec_witness: Vec<FieldElement<8>> = result
                .iter()
                .map(|a| {
                    let a: [u8; 8] = a.to_le_bytes().try_into().unwrap();
                    a.into()
                })
                .collect();
            Ok(wtns_from_witness2(vec_witness, prime))
        }
        32 => {
            let vec_witness: Vec<FieldElement<32>> = result
                .iter()
                .map(|a| {
                    let a: [u8; 32] = a.to_le_bytes().try_into().unwrap();
                    a.into()
                })
                .collect();
            Ok(wtns_from_witness2(vec_witness, prime))
        }
        _ => {
            Err(CompilationError::WitnessSignalNotSet)
        }
    }

}

fn disassemble<T: FieldOps>(templates: &[vm2::Template]) {
    for t in templates.iter() {
        println!("[begin]Template: {}", t.name);
        let mut ip: usize = 0;
        while ip < t.code.len() {
            ip = disassemble_instruction::<T>(&t.code, ip, &t.name);
        }
        println!("[end]")
    }
}

fn compile<T>(
    tree: &ast::AST, sym_file: &str) -> Result<Circuit<T>, Box<dyn Error>>
where
    T: FieldOps {

    let mut templates = Vec::new();

    for t in tree.templates.iter() {
        let compiled_template = compile_template::<T>(t);
        templates.push(compiled_template);
        // println!("Template: {}", t.name);
        // println!("Compiled code len: {}", compiled_template.code.len());
    }

    let mut main_template_id = None;
    for (i, t) in templates.iter().enumerate() {
        if t.name == tree.start {
            main_template_id = Some(i)
        }
    }

    let main_template_id = main_template_id
        .ok_or(CompilationError::MainTemplateIDNotFound)?;

    let prime = T::from_le_bytes(&tree.prime.to_le_bytes())?;

    Ok(Circuit {
        main_template_id,
        templates,
        prime,
        witness: tree.witness.clone(),
        input_signals_info: input_signals_info(sym_file, main_template_id)?
    })
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
            ctx.code.push(OpCode::PushFf as u8);
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
            ctx.code.push(OpCode::StoreVariableFf as u8);
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
        vars_i64_num: ctx.i64_variable_indexes.len(),
        vars_ff_num: ctx.ff_variable_indexes.len(),
    }
}

#[cfg(test)]
mod tests {
    use circom_witnesscalc::ast::{Assignment, Expr, FfOperand, I64Operand, TemplateInstruction};
    use super::*;

    #[test]
    fn test_example() {
        assert!(true);
    }

    #[test]
    fn test_compile_template() {
        let ast_tmpl = ast::Template {
            name: "Multiplier_0".to_string(),
            outputs: vec![],
            inputs: vec![],
            signals_num: 0,
            components: vec![],
            body: vec![
                TemplateInstruction::Assignment(
                    Assignment{
                        dest: "x_0".to_string(),
                        value: Expr::GetSignal(I64Operand::Literal(1)),
                    }
                ),
                TemplateInstruction::Assignment(
                    Assignment{
                        dest: "x_1".to_string(),
                        value: Expr::GetSignal(I64Operand::Literal(2)),
                    }
                ),
                TemplateInstruction::Assignment(
                    Assignment{
                        dest: "x_2".to_string(),
                        value: Expr::FfMul(
                            FfOperand::Variable("x_0".to_string()),
                            FfOperand::Variable("x_1".to_string()))}),
                TemplateInstruction::Assignment(
                    Assignment{
                        dest: "x_3".to_string(),
                        value: Expr::FfAdd(
                            FfOperand::Variable("x_2".to_string()),
                            FfOperand::Literal(BigUint::from(2u32)))}),
                TemplateInstruction::Statement(
                    Statement::SetSignal {
                        idx: I64Operand::Literal(0),
                        value: FfOperand::Variable("x_3".to_string())})
            ],
        };
        let vm_tmpl = compile_template::<U254>(&ast_tmpl);
        disassemble::<U254>(&vec![vm_tmpl]);
    }
}