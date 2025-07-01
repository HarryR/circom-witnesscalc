use std::{env, fs, process};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, Write};
use std::path::Path;
use num_bigint::BigUint;
use num_traits::{Num, ToBytes};
use wtns_file::FieldElement;
use circom_witnesscalc::{ast, vm2, wtns_from_witness2};
use circom_witnesscalc::field::{bn254_prime, Field, FieldOperations, FieldOps, U254};
use circom_witnesscalc::parser::parse;
use circom_witnesscalc::vm2::{disassemble_instruction, execute, Circuit, Component, OpCode};

struct WantWtns {
    wtns_file: String,
    inputs_file: String,
}

struct Args {
    cvm_file: String,
    output_file: String,
    want_wtns: Option<WantWtns>,
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
    #[error("jump offset is too large")]
    JumpOffsetIsTooLarge,
    #[error("[assertion] Loop control stack is empty")]
    LoopControlJumpsEmpty
}

#[derive(Debug, thiserror::Error)]
enum RuntimeError {
    #[error("incorrect inputs json file: `{0}`")]
    InvalidSignalsJson(String)
}

fn parse_args() -> Args {
    let mut cvm_file: Option<String> = None;
    let mut output_file: Option<String> = None;
    let mut wtns_file: Option<String> = None;
    let mut inputs_file: Option<String> = None;
    let mut sym_file: Option<String> = None;

    let args: Vec<String> = env::args().collect();

    let usage = |err_msg: &str| -> ! {
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
        eprintln!("    -h | --help       Display this help message");
        eprintln!("    --wtns            If file is provided, the witness will be calculated and saved in this file. Inputs file MUST be provided as well.");
        eprintln!("    --inputs          File with inputs for the circuit. Required if --wtns is provided.");
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
        } else if args[i] == "--inputs" {
            i += 1;
            if i >= args.len() {
                usage("missing argument for --inputs");
            }
            if inputs_file.is_some() {
                usage("multiple inputs files");
            }
            inputs_file = Some(args[i].clone());
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

    let want_wtns: Option<WantWtns> = match (inputs_file, wtns_file) {
        (Some(inputs_file), Some(wtns_file)) => {
            Some(WantWtns{ wtns_file, inputs_file })
        }
        (None, None) => None,
        (Some(_), None) => {
            usage("inputs file is provided, but witness file is not");
        }
        (None, Some(_)) => {
            usage("witness file is provided, but inputs file is not");
        }
    };

    Args {
        cvm_file: cvm_file.unwrap_or_else(|| { usage("missing CVM file") }),
        output_file: output_file.unwrap_or_else(|| { usage("missing output file") }),
        want_wtns,
        sym_file: sym_file.unwrap_or_else(|| { usage("missing SYM file") }),
    }
}

fn main() {
    let args = parse_args();

    let program_text = fs::read_to_string(&args.cvm_file).unwrap();
    let program = match parse(&program_text) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            process::exit(1);
        }
    };

    println!("number of templates: {}", program.templates.len());
    let bn254 = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
    if program.prime == bn254 {
        let ff = Field::new(bn254_prime);
        let circuit = compile(&ff, &program, &args.sym_file).unwrap();
        let mut component_tree = build_component_tree(
            &program.templates, circuit.main_template_id);
        disassemble::<U254>(&circuit.templates);
        if args.want_wtns.is_some() {
            calculate_witness(
                &circuit, &mut component_tree, args.want_wtns.unwrap())
                .unwrap();
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

/// Create a component tree and returns the component and the number of signals
/// of self and all its children
fn create_component(
    templates: &[ast::Template], template_id: usize,
    signals_start: usize) -> (Component, usize) {

    let t = &templates[template_id];
    let mut next_signal_start = signals_start + t.signals_num;
    let mut components = Vec::with_capacity(t.components.len());
    for cmp_tmpl_id in t.components.iter() {
        components.push(match cmp_tmpl_id {
            None => None,
            Some( tmpl_id ) => {
                let (c, signals_num) = create_component(
                    templates, *tmpl_id, next_signal_start);
                next_signal_start += signals_num;
                Some(Box::new(c))
            }
        });
    }
    (
        Component {
            signals_start,
            template_id,
            components,
            number_of_inputs: t.outputs.len(),
        },
        next_signal_start - signals_start
    )
}

fn build_component_tree(
    templates: &[ast::Template], main_template_id: usize) -> Component {

    create_component(templates, main_template_id, 1).0
}

fn calculate_witness<T: FieldOps>(
    circuit: &Circuit<T>, component_tree: &mut Component,
    want_wtns: WantWtns) -> Result<(), Box<dyn Error>> {

    let mut signals = init_signals(
        &want_wtns.inputs_file, circuit.signals_num, &circuit.field,
        &circuit.input_signals_info)?;
    execute(&circuit.templates, &mut signals, &circuit.field, component_tree)?;
    let wtns_data = witness(
        &signals, &circuit.witness, circuit.field.prime)?;

    let mut file = File::create(Path::new(&want_wtns.wtns_file))?;
    file.write_all(&wtns_data)?;
    file.flush()?;
    println!("Witness saved to {}", want_wtns.wtns_file);
    Ok(())
}

fn parse_signals_json<T: FieldOps, F>(
    inputs_data: &[u8], ff: &F) -> Result<HashMap<String, T>, Box<dyn Error>>
where
    for <'a> &'a F: FieldOperations<Type = T> {

    let v: serde_json::Value = serde_json::from_slice(inputs_data)?;
    let mut records: HashMap<String, T> = HashMap::new();
    visit_inputs_json("main", &v, &mut records, ff)?;
    Ok(records)
}

fn visit_inputs_json<T: FieldOps, F>(
    prefix: &str, v: &serde_json::Value, records: &mut HashMap<String, T>,
    ff: &F) -> Result<(), Box<dyn Error>>
where
    for <'a> &'a F: FieldOperations<Type = T> {

    match v {
        serde_json::Value::Null => return Err(Box::new(
            RuntimeError::InvalidSignalsJson(
                format!("unexpected null value at path {}", prefix)))),
        serde_json::Value::Bool(b) => {
            let b = if *b { T::one() } else { T::zero() };
            records.insert(prefix.to_string(), b);
        },
        serde_json::Value::Number(n) => {
            let v = if n.is_u64() {
                let n = n.as_u64().unwrap();
                ff.parse_le_bytes(n.to_le_bytes().as_slice())?
            } else if n.is_i64() {
                let n = n.as_i64().unwrap();
                ff.parse_str(&n.to_string())?
            } else {
                return Err(Box::new(RuntimeError::InvalidSignalsJson(
                    format!("invalid number at path {}: {}", prefix, n))));
            };
            records.insert(prefix.to_string(), v);
        },
        serde_json::Value::String(s) => {
            records.insert(prefix.to_string(), ff.parse_str(s)?);
        },
        serde_json::Value::Array(vs) => {
            for (i, v) in vs.iter().enumerate() {
                let new_prefix = format!("{}[{}]", prefix, i);
                visit_inputs_json(&new_prefix, v, records, ff)?;
            }
        },
        serde_json::Value::Object(o) => {
            for (k, v) in o.iter() {
                let new_prefix = prefix.to_string() + "." + k;
                visit_inputs_json(&new_prefix, v, records, ff)?;
            }
        },
    };

    Ok(())
}

fn init_signals<T: FieldOps, F>(
    inputs_file: &str, signals_num: usize, ff: &F,
    input_signals_info: &HashMap<String, usize>) -> Result<Vec<Option<T>>, Box<dyn Error>>
where
    for <'a> &'a F: FieldOperations<Type = T> {

    let mut signals = vec![None; signals_num];
    signals[0] = Some(T::one());

    let inputs_data = fs::read_to_string(inputs_file)?;
    let input_signals = parse_signals_json(inputs_data.as_bytes(), ff)?;
    for (path, value) in input_signals.iter() {
        match input_signals_info.get(path) {
            None => {
                if path.ends_with("[0]") {
                    let path = path.trim_end_matches("[0]");
                    if let Some(signal_idx) = input_signals_info.get(path) {
                        signals[*signal_idx] = Some(*value);
                        continue;
                    }
                }
                return Err(Box::new(
                    RuntimeError::InvalidSignalsJson(
                        format!("signal {} is not found in SYM file", path))))
            },
            Some(signal_idx) => signals[*signal_idx] = Some(*value),
        }
    }

    Ok(signals)
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

fn compile<T: FieldOps>(
    ff: &Field<T>, tree: &ast::AST, sym_file: &str) -> Result<Circuit<T>, Box<dyn Error>>
where {

    let mut templates = Vec::new();

    for t in tree.templates.iter() {
        let compiled_template = compile_template(t, ff)?;
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

    Ok(Circuit {
        main_template_id,
        templates,
        field: ff.clone(),
        witness: tree.witness.clone(),
        input_signals_info: input_signals_info(sym_file, main_template_id)?,
        signals_num: tree.signals,
    })
}

struct TemplateCompilationContext {
    code: Vec<u8>,
    ff_variable_indexes: HashMap<String, i64>,
    i64_variable_indexes: HashMap<String, i64>,
    // Stack of loop render frames.
    // * The first element of the tuple is the position of the first loop body
    // instruction (the continue statement).
    // * The second element is a vector of indexes where to inject the
    // position after loop body (the break statement)
    loop_control_jumps: Vec<(usize, Vec<usize>)>,
}

impl TemplateCompilationContext {
    fn new() -> Self {
        Self {
            code: vec![],
            ff_variable_indexes: HashMap::new(),
            i64_variable_indexes: HashMap::new(),
            loop_control_jumps: vec![],
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

fn i64_expression(
    ctx: &mut TemplateCompilationContext,
    expr: &ast::I64Expr) -> Result<(), Box<dyn Error>> {
    
    match expr {
        ast::I64Expr::Variable(var_name) => {
            let var_idx = ctx.get_i64_variable_index(var_name);
            ctx.code.push(OpCode::LoadVariableI64 as u8);
            ctx.code.extend_from_slice(var_idx.to_le_bytes().as_slice());
        }
        ast::I64Expr::Literal(value) => {
            ctx.code.push(OpCode::PushI64 as u8);
            ctx.code.extend_from_slice(value.to_le_bytes().as_slice());
        }
        ast::I64Expr::Add(lhs, rhs) => {
            i64_expression(ctx, rhs)?;
            i64_expression(ctx, lhs)?;
            ctx.code.push(OpCode::OpI64Add as u8);
        }
        ast::I64Expr::Sub(lhs, rhs) => {
            i64_expression(ctx, rhs)?;
            i64_expression(ctx, lhs)?;
            ctx.code.push(OpCode::OpI64Sub as u8);
        }
        ast::I64Expr::Mul(_lhs, _rhs) => {
            // TODO: Implement i64.mul operation
            // This requires adding a new OpCode for i64 multiplication
            return Err("i64.mul operation not yet implemented in VM".into());
        }
        ast::I64Expr::Lte(_lhs, _rhs) => {
            // TODO: Implement i64.le operation
            // This requires adding a new OpCode for i64 less-than-or-equal comparison
            return Err("i64.le operation not yet implemented in VM".into());
        }
        ast::I64Expr::Load(_addr) => {
            // TODO: Implement i64.load memory operation
            // This requires adding a new OpCode for memory load operations
            return Err("i64.load operation not yet implemented in VM".into());
        }
        ast::I64Expr::Wrap(_ff_expr) => {
            // TODO: Implement i64.wrap_ff operation
            // This requires adding a new OpCode for wrapping FF values to i64
            return Err("i64.wrap_ff operation not yet implemented in VM".into());
        }
    }
    Ok(())
}

fn ff_expression<F>(
    ctx: &mut TemplateCompilationContext, ff: &F,
    expr: &ast::FfExpr) -> Result<(), Box<dyn Error>>
where
    for <'a> &'a F: FieldOperations {

    match expr {
        ast::FfExpr::GetSignal(operand) => {
            operand_i64(ctx, operand);
            ctx.code.push(OpCode::LoadSignal as u8);
        }
        ast::FfExpr::GetCmpSignal{ cmp_idx, sig_idx } => {
            operand_i64(ctx, cmp_idx);
            operand_i64(ctx, sig_idx);
            ctx.code.push(OpCode::LoadCmpSignal as u8);
        }
        ast::FfExpr::FfMul(lhs, rhs) => {
            ff_expression(ctx, ff, rhs)?;
            ff_expression(ctx, ff, lhs)?;
            ctx.code.push(OpCode::OpMul as u8);
        },
        ast::FfExpr::FfAdd(lhs, rhs) => {
            ff_expression(ctx, ff, rhs)?;
            ff_expression(ctx, ff, lhs)?;
            ctx.code.push(OpCode::OpAdd as u8);
        },
        ast::FfExpr::FfNeq(lhs, rhs) => {
            ff_expression(ctx, ff, rhs)?;
            ff_expression(ctx, ff, lhs)?;
            ctx.code.push(OpCode::OpNeq as u8);
        },
        ast::FfExpr::FfDiv(lhs, rhs) => {
            ff_expression(ctx, ff, rhs)?;
            ff_expression(ctx, ff, lhs)?;
            ctx.code.push(OpCode::OpDiv as u8);
        },
        ast::FfExpr::FfSub(lhs, rhs) => {
            ff_expression(ctx, ff, rhs)?;
            ff_expression(ctx, ff, lhs)?;
            ctx.code.push(OpCode::OpSub as u8);
        },
        ast::FfExpr::FfEq(lhs, rhs) => {
            ff_expression(ctx, ff, rhs)?;
            ff_expression(ctx, ff, lhs)?;
            ctx.code.push(OpCode::OpEq as u8);
        },
        ast::FfExpr::FfEqz(lhs) => {
            ff_expression(ctx, ff, lhs)?;
            ctx.code.push(OpCode::OpEqz as u8);
        },
        ast::FfExpr::Variable( var_name ) => {
            let var_idx = ctx.get_ff_variable_index(var_name);
            ctx.code.push(OpCode::LoadVariableFf as u8);
            ctx.code.extend_from_slice(var_idx.to_le_bytes().as_slice());
        },
        ast::FfExpr::Literal(v) => {
            ctx.code.push(OpCode::PushFf as u8);
            let x = ff.parse_le_bytes(v.to_le_bytes().as_slice())?;
            ctx.code.extend_from_slice(x.to_le_bytes().as_slice());
        },
        ast::FfExpr::Load(_idx) => {
            // TODO: Implement ff.load memory operation
            // This requires adding a new OpCode for memory load operations
            return Err("ff.load operation not yet implemented in VM".into());
        },
        ast::FfExpr::Lt(_lhs, _rhs) => {
            // TODO: Implement ff.lt comparison operation
            // This requires adding a new OpCode for less-than comparison
            return Err("ff.lt operation not yet implemented in VM".into());
        },
    };
    Ok(())
}

fn instruction<F>(
    ctx: &mut TemplateCompilationContext, ff: &F,
    inst: &ast::TemplateInstruction) -> Result<(), Box<dyn Error>>
where
    for <'a> &'a F: FieldOperations {

    match inst {
        ast::TemplateInstruction::FfAssignment(assignment) => {
            ff_expression(ctx, ff, &assignment.value)?;
            ctx.code.push(OpCode::StoreVariableFf as u8);
            let var_idx = ctx.get_ff_variable_index(&assignment.dest);
            ctx.code.extend_from_slice(var_idx.to_le_bytes().as_slice());
        }
        ast::TemplateInstruction::I64Assignment(assignment) => {
            i64_expression(ctx, &assignment.value)?;
            ctx.code.push(OpCode::StoreVariableI64 as u8);
            let var_idx = ctx.get_i64_variable_index(&assignment.dest);
            ctx.code.extend_from_slice(var_idx.to_le_bytes().as_slice());
        }
        ast::TemplateInstruction::Statement(statement) => {
            match statement {
                ast::Statement::SetSignal { idx, value } => {
                    ff_expression(ctx, ff, value)?;
                    operand_i64(ctx, idx);
                    ctx.code.push(OpCode::StoreSignal as u8);
                },
                ast::Statement::FfStore { idx: _, value: _ } => {
                    // TODO: Implement ff.store memory operation
                    // This requires adding a new OpCode for memory store operations
                    return Err("ff.store operation not yet implemented in VM".into());
                },
                ast::Statement::SetCmpSignalRun { cmp_idx, sig_idx, value } => {
                    operand_i64(ctx, cmp_idx);
                    operand_i64(ctx, sig_idx);
                    ff_expression(ctx, ff, value)?;
                    ctx.code.push(OpCode::StoreCmpSignalAndRun as u8);
                },
                ast::Statement::SetCmpInput { cmp_idx, sig_idx, value } => {
                    i64_expression(ctx, cmp_idx)?;
                    i64_expression(ctx, sig_idx)?;
                    ff_expression(ctx, ff, value)?;
                    ctx.code.push(OpCode::StoreCmpInput as u8);
                },
                ast::Statement::Branch { condition, if_block, else_block } => {
                    let else_jump_offset = match condition {
                        ast::Expr::Ff(expr) => {
                            ff_expression(ctx, ff, expr)?;
                            pre_emit_jump_if_false(&mut ctx.code, true)
                        },
                        ast::Expr::I64(expr) => {
                            i64_expression(ctx, expr)?;
                            pre_emit_jump_if_false(&mut ctx.code, false)
                        },
                    };

                    block(ctx, ff, if_block)?;

                    let to = ctx.code.len();
                    patch_jump(&mut ctx.code, else_jump_offset, to)?;

                    let end_jump_offset = pre_emit_jump(&mut ctx.code);
                    block(ctx, ff, else_block)?;

                    let to = ctx.code.len();
                    patch_jump(&mut ctx.code, end_jump_offset, to)?;
                },
                ast::Statement::Loop( loop_block ) => {
                    // Start of loop
                    let loop_start = ctx.code.len();
                    ctx.loop_control_jumps = vec![(loop_start, vec![])];

                    // Compile loop body
                    block(ctx, ff, loop_block)?;

                    let to = ctx.code.len();
                    let (_, break_jumps) = ctx.loop_control_jumps.pop()
                        .ok_or(CompilationError::LoopControlJumpsEmpty)?;
                    for break_jump_offset in break_jumps {
                        patch_jump(&mut ctx.code, break_jump_offset, to)?;
                    }
                },
                ast::Statement::Break => {
                    let jump_offset = pre_emit_jump(&mut ctx.code);
                    if let Some((_, break_jumps)) = ctx.loop_control_jumps.last_mut() {
                        break_jumps.push(jump_offset);
                    } else {
                        return Err(Box::new(CompilationError::LoopControlJumpsEmpty));
                    }
                },
                ast::Statement::Continue => {
                    let jump_offset = pre_emit_jump(&mut ctx.code);
                    patch_jump(&mut ctx.code, jump_offset, ctx.loop_control_jumps.last()
                        .ok_or(CompilationError::LoopControlJumpsEmpty)?.0)?;
                },
                ast::Statement::Error { code } => {
                    operand_i64(ctx, code);
                    ctx.code.push(OpCode::Error as u8);
                },
                ast::Statement::FfMReturn { dst, src, size } => {
                    // Push operands in reverse order so they are popped in correct order
                    // The VM expects: stack[-2]=dst, stack[-1]=src, stack[0]=size
                    operand_i64(ctx, dst);
                    operand_i64(ctx, src);
                    operand_i64(ctx, size);
                    ctx.code.push(OpCode::FfMReturn as u8);
                },
                ast::Statement::FfMCall { name: _name, args } => {
                    // Emit the FfMCall opcode
                    ctx.code.push(OpCode::FfMCall as u8);
                    
                    // TODO: Look up function by name to get its index
                    // For now, use a placeholder index of 0
                    let func_idx: u32 = 0;
                    ctx.code.extend_from_slice(&func_idx.to_le_bytes());
                    
                    // Emit argument count
                    ctx.code.push(args.len() as u8);
                    
                    // Emit each argument
                    for arg in args {
                        match arg {
                            ast::CallArgument::I64Literal(value) => {
                                ctx.code.push(0); // arg type 0 = i64 literal
                                ctx.code.extend_from_slice(&value.to_le_bytes());
                            }
                            ast::CallArgument::FfLiteral(value) => {
                                ctx.code.push(1); // arg type 1 = ff literal
                                let x = ff.parse_le_bytes(value.to_le_bytes().as_slice())?;
                                ctx.code.extend_from_slice(x.to_le_bytes().as_slice());
                            }
                            ast::CallArgument::I64Memory { addr, size } => {
                                ctx.code.push(2); // arg type 2 = i64 memory
                                match addr {
                                    ast::I64Operand::Literal(v) => {
                                        ctx.code.extend_from_slice(&v.to_le_bytes());
                                    }
                                    ast::I64Operand::Variable(var_name) => {
                                        let var_idx = ctx.get_i64_variable_index(var_name);
                                        ctx.code.extend_from_slice(&var_idx.to_le_bytes());
                                    }
                                }
                                match size {
                                    ast::I64Operand::Literal(v) => {
                                        ctx.code.extend_from_slice(&v.to_le_bytes());
                                    }
                                    ast::I64Operand::Variable(var_name) => {
                                        let var_idx = ctx.get_i64_variable_index(var_name);
                                        ctx.code.extend_from_slice(&var_idx.to_le_bytes());
                                    }
                                }
                            }
                            ast::CallArgument::FfMemory { addr, size } => {
                                ctx.code.push(3); // arg type 3 = ff memory
                                match addr {
                                    ast::I64Operand::Literal(v) => {
                                        ctx.code.extend_from_slice(&v.to_le_bytes());
                                    }
                                    ast::I64Operand::Variable(var_name) => {
                                        let var_idx = ctx.get_i64_variable_index(var_name);
                                        ctx.code.extend_from_slice(&var_idx.to_le_bytes());
                                    }
                                }
                                match size {
                                    ast::I64Operand::Literal(v) => {
                                        ctx.code.extend_from_slice(&v.to_le_bytes());
                                    }
                                    ast::I64Operand::Variable(var_name) => {
                                        let var_idx = ctx.get_i64_variable_index(var_name);
                                        ctx.code.extend_from_slice(&var_idx.to_le_bytes());
                                    }
                                }
                            }
                        }
                    }
                    todo!();
                }
            }
        }
    };
    Ok(())
}

fn block<F>(
    ctx: &mut TemplateCompilationContext, ff: &F,
    instructions: &[ast::TemplateInstruction]) -> Result<(), Box<dyn Error>>
where
    for <'a> &'a F: FieldOperations {

    for inst in instructions {
        instruction(ctx, ff, inst)?;
    }

    Ok(())
}

fn calc_jump_offset(from: usize, to: usize) -> Result<i32, CompilationError> {
    let from: i64 = from.try_into()
        .map_err(|_| CompilationError::JumpOffsetIsTooLarge)?;
    let to: i64 = to.try_into()
        .map_err(|_| CompilationError::JumpOffsetIsTooLarge)?;

    (to - from).try_into()
        .map_err(|_| CompilationError::JumpOffsetIsTooLarge)
}


/// We expect the jump offset located at `jump_offset_addr` to be 4 bytes long.
/// The jump offset is calculated as `to - jump_offset_addr - 4`.
fn patch_jump(
    code: &mut [u8], jump_offset_addr: usize,
    to: usize) -> Result<(), CompilationError> {

    let offset = calc_jump_offset(jump_offset_addr + 4, to)?;
    code[jump_offset_addr..jump_offset_addr+4].copy_from_slice(offset.to_le_bytes().as_ref());
    Ok(())
}


fn pre_emit_jump_if_false(code: &mut Vec<u8>, is_ff: bool) -> usize {
    if is_ff {
        code.push(OpCode::JumpIfFalseFf as u8);
    } else {
        code.push(OpCode::JumpIfFalseI64 as u8);
    }
    for _ in 0..4 { code.push(0xffu8); }
    code.len() - 4
}

fn pre_emit_jump(code: &mut Vec<u8>) -> usize {
    code.push(OpCode::Jump as u8);
    for _ in 0..4 { code.push(0xffu8); }
    code.len() - 4
}


fn compile_template<F>(t: &ast::Template, ff: &F) -> Result<vm2::Template, Box<dyn Error>>
where
    for <'a> &'a F: FieldOperations {

    let mut ctx = TemplateCompilationContext::new();
    for i in &t.body {
        instruction(&mut ctx, ff, i)?;
    }

    Ok(vm2::Template {
        name: t.name.clone(),
        code: ctx.code,
        vars_i64_num: ctx.i64_variable_indexes.len(),
        vars_ff_num: ctx.ff_variable_indexes.len(),
    })
}

#[cfg(test)]
mod tests {
    use circom_witnesscalc::ast::{FfAssignment, FfExpr, I64Operand, TemplateInstruction, Signal};
    use circom_witnesscalc::field::{bn254_prime, Field};
    use super::*;

    #[test]
    fn test_example() {
        // Placeholder test
    }

    #[test]
    fn test_build_component_tree() {
        // Create leaf templates with no components
        let template1 = ast::Template {
            name: "Leaf1".to_string(),
            outputs: vec![Signal::Ff(vec![1])],
            inputs: vec![Signal::Ff(vec![1])],
            signals_num: 3,
            components: vec![],
            body: vec![],
        };

        let template2 = ast::Template {
            name: "Leaf2".to_string(),
            outputs: vec![Signal::Ff(vec![1])],
            inputs: vec![Signal::Ff(vec![1])],
            signals_num: 3,
            components: vec![],
            body: vec![],
        };

        let template3 = ast::Template {
            name: "Leaf3".to_string(),
            outputs: vec![Signal::Ff(vec![1])],
            inputs: vec![Signal::Ff(vec![1])],
            signals_num: 3,
            components: vec![],
            body: vec![],
        };

        let template4 = ast::Template {
            name: "Leaf4".to_string(),
            outputs: vec![Signal::Ff(vec![1])],
            inputs: vec![Signal::Ff(vec![1])],
            signals_num: 3,
            components: vec![],
            body: vec![],
        };

        // Create middle-level templates, each with two children
        // First middle template has two children
        let template5 = ast::Template {
            name: "Middle1".to_string(),
            outputs: vec![Signal::Ff(vec![1])],
            inputs: vec![Signal::Ff(vec![1])],
            signals_num: 4,
            components: vec![Some(0), Some(1)], // References to template1 and template2
            body: vec![],
        };

        // Second middle template has one child and one None
        let template6 = ast::Template {
            name: "Middle2".to_string(),
            outputs: vec![Signal::Ff(vec![1])],
            inputs: vec![Signal::Ff(vec![1])],
            signals_num: 4,
            components: vec![Some(2), None, Some(3)], // References to template3, None, and template4
            body: vec![],
        };

        // Create root template with two children
        let template7 = ast::Template {
            name: "Root".to_string(),
            outputs: vec![Signal::Ff(vec![1])],
            inputs: vec![Signal::Ff(vec![1])],
            signals_num: 5,
            components: vec![Some(4), Some(5)], // References to template5 and template6
            body: vec![],
        };

        let templates = vec![template1, template2, template3, template4, template5, template6, template7];

        // Build component tree with template7 (Root) as the main template
        let component_tree = build_component_tree(&templates, 6);

        // Verify the structure of the root component
        assert_eq!(component_tree.signals_start, 1);
        assert_eq!(component_tree.template_id, 6);
        assert_eq!(component_tree.number_of_inputs, 1);
        assert_eq!(component_tree.components.len(), 2);

        // Verify the first child component (Middle1)
        let middle1 = component_tree.components[0].as_ref().unwrap();
        assert_eq!(middle1.signals_start, 6); // 1 (start) + 5 (signals_num of root)
        assert_eq!(middle1.template_id, 4);
        assert_eq!(middle1.number_of_inputs, 1);
        assert_eq!(middle1.components.len(), 2);

        // Verify the second child component (Middle2)
        let middle2 = component_tree.components[1].as_ref().unwrap();
        assert_eq!(middle2.signals_start, 16); // 6 (start of middle1) + 4 (signals_num of middle1) + 3 (signals_num of leaf1) + 3 (signals_num of leaf2)
        assert_eq!(middle2.template_id, 5);
        assert_eq!(middle2.number_of_inputs, 1);
        assert_eq!(middle2.components.len(), 3);

        // Verify Middle2 has a None component
        assert!(middle2.components[1].is_none());

        // Verify the leaf components of Middle1
        let leaf1 = middle1.components[0].as_ref().unwrap();
        assert_eq!(leaf1.signals_start, 10); // 6 (start of middle1) + 4 (signals_num of middle1)
        assert_eq!(leaf1.template_id, 0);
        assert_eq!(leaf1.number_of_inputs, 1);
        assert_eq!(leaf1.components.len(), 0);

        let leaf2 = middle1.components[1].as_ref().unwrap();
        assert_eq!(leaf2.signals_start, 13); // 10 (start of leaf1) + 3 (signals_num of leaf1)
        assert_eq!(leaf2.template_id, 1);
        assert_eq!(leaf2.number_of_inputs, 1);
        assert_eq!(leaf2.components.len(), 0);

        // Verify the leaf components of Middle2
        let leaf3 = middle2.components[0].as_ref().unwrap();
        assert_eq!(leaf3.signals_start, 20); // 16 (start of middle2) + 4 (signals_num of middle2)
        assert_eq!(leaf3.template_id, 2);
        assert_eq!(leaf3.number_of_inputs, 1);
        assert_eq!(leaf3.components.len(), 0);

        let leaf4 = middle2.components[2].as_ref().unwrap();
        assert_eq!(leaf4.signals_start, 23); // 20 (start of leaf3) + 3 (signals_num of leaf3)
        assert_eq!(leaf4.template_id, 3);
        assert_eq!(leaf4.number_of_inputs, 1);
        assert_eq!(leaf4.components.len(), 0);
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
                TemplateInstruction::FfAssignment(
                    FfAssignment {
                        dest: "x_0".to_string(),
                        value: FfExpr::GetSignal(I64Operand::Literal(1)),
                    }
                ),
                TemplateInstruction::FfAssignment(
                    FfAssignment {
                        dest: "x_1".to_string(),
                        value: FfExpr::GetSignal(I64Operand::Literal(2)),
                    }
                ),
                TemplateInstruction::FfAssignment(
                    FfAssignment {
                        dest: "x_2".to_string(),
                        value: FfExpr::FfMul(
                            Box::new(FfExpr::Variable("x_0".to_string())),
                            Box::new(FfExpr::Variable("x_1".to_string())))}),
                TemplateInstruction::FfAssignment(
                    FfAssignment {
                        dest: "x_3".to_string(),
                        value: FfExpr::FfAdd(
                            Box::new(FfExpr::Variable("x_2".to_string())),
                            Box::new(FfExpr::Literal(BigUint::from(2u32))))}),
                TemplateInstruction::Statement(
                    ast::Statement::SetSignal {
                        idx: I64Operand::Literal(0),
                        value: FfExpr::Variable("x_3".to_string())})
            ],
        };
        let ff = Field::new(bn254_prime);
        let vm_tmpl = compile_template(&ast_tmpl, &ff).unwrap();
        disassemble::<U254>(&[vm_tmpl]);
    }

    #[test]
    fn test_parse_signals_json() {
        let ff = Field::new(bn254_prime);

        // bools
        let i = r#"
{
  "a": true,
  "b": false,
  "c": 100500
}"#;
        let result = parse_signals_json(i.as_bytes(), &ff).unwrap();
        let mut want: HashMap<String, U254> = HashMap::new();
        want.insert("main.a".to_string(), U254::from_str("1").unwrap());
        want.insert("main.b".to_string(), U254::from_str("0").unwrap());
        want.insert("main.c".to_string(), U254::from_str("100500").unwrap());
        assert_eq!(want, result);

        // embedded objects
        let i = r#"{ "a": { "b": true } }"#;
        let result = parse_signals_json(i.as_bytes(), &ff).unwrap();
        let mut want: HashMap<String, U254> = HashMap::new();
        want.insert("main.a.b".to_string(), U254::from_str("1").unwrap());
        assert_eq!(want, result);

        // null error
        let i = r#"{ "a": { "b": null } }"#;
        let result = parse_signals_json(i.as_bytes(), &ff);
        let binding = result.unwrap_err();
        let err = binding.downcast_ref::<RuntimeError>().unwrap();
        assert!(matches!(err, RuntimeError::InvalidSignalsJson(x) if x == "unexpected null value at path main.a.b"));

        // Negative number
        let i = r#"{ "a": { "b": -4 } }"#;
        let result = parse_signals_json(i.as_bytes(), &ff).unwrap();
        let mut want: HashMap<String, U254> = HashMap::new();
        want.insert("main.a.b".to_string(), U254::from_str("21888242871839275222246405745257275088548364400416034343698204186575808495613").unwrap());
        assert_eq!(want, result);

        // Float number error
        let i = r#"{ "a": { "b": 8.3 } }"#;
        let result = parse_signals_json(i.as_bytes(), &ff);
        let binding = result.unwrap_err();
        let err = binding.downcast_ref::<RuntimeError>().unwrap();
        let msg = err.to_string();
        assert!(matches!(err, RuntimeError::InvalidSignalsJson(x) if x == "invalid number at path main.a.b: 8.3"), "{}", msg);

        // string
        let i = r#"{ "a": { "b": "8" } }"#;
        let result = parse_signals_json(i.as_bytes(), &ff).unwrap();
        let mut want: HashMap<String, U254> = HashMap::new();
        want.insert("main.a.b".to_string(), U254::from_str("8").unwrap());
        assert_eq!(want, result);

        // array
        let i = r#"{ "a": { "b": ["8", 2, 3] } }"#;
        let result = parse_signals_json(i.as_bytes(), &ff).unwrap();
        let mut want: HashMap<String, U254> = HashMap::new();
        want.insert("main.a.b[0]".to_string(), U254::from_str("8").unwrap());
        want.insert("main.a.b[1]".to_string(), U254::from_str("2").unwrap());
        want.insert("main.a.b[2]".to_string(), U254::from_str("3").unwrap());
        assert_eq!(want, result);

        // buses and arrays
        let i = r#"{
  "a": ["300", 3, "8432", 3, 2],
  "inB": "100500",
  "v": {
    "v": [
      {
        "start": {"x": 3, "y": 5},
        "end": {"x": 6, "y": 7}
      },
      {
        "start": {"x": 8, "y": 9},
        "end": {"x": 10, "y": 11}
      }
    ]
  }
}"#;
        let result = parse_signals_json(i.as_bytes(), &ff).unwrap();
        let mut want: HashMap<String, U254> = HashMap::new();
        want.insert("main.a[0]".to_string(), U254::from_str("300").unwrap());
        want.insert("main.a[1]".to_string(), U254::from_str("3").unwrap());
        want.insert("main.a[2]".to_string(), U254::from_str("8432").unwrap());
        want.insert("main.a[3]".to_string(), U254::from_str("3").unwrap());
        want.insert("main.a[4]".to_string(), U254::from_str("2").unwrap());
        want.insert("main.inB".to_string(), U254::from_str("100500").unwrap());
        want.insert("main.v.v[0].start.x".to_string(), U254::from_str("3").unwrap());
        want.insert("main.v.v[0].start.y".to_string(), U254::from_str("5").unwrap());
        want.insert("main.v.v[0].end.x".to_string(), U254::from_str("6").unwrap());
        want.insert("main.v.v[0].end.y".to_string(), U254::from_str("7").unwrap());
        want.insert("main.v.v[1].start.x".to_string(), U254::from_str("8").unwrap());
        want.insert("main.v.v[1].start.y".to_string(), U254::from_str("9").unwrap());
        want.insert("main.v.v[1].end.x".to_string(), U254::from_str("10").unwrap());
        want.insert("main.v.v[1].end.y".to_string(), U254::from_str("11").unwrap());
        assert_eq!(want, result);
    }
}
