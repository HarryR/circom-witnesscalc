use num_bigint::BigUint;
use winnow::ascii::{digit1, space0, space1, line_ending, alpha1};
use winnow::error::{ContextError, ErrMode, ParseError, ParserError, StrContext, StrContextValue};
use winnow::{ModalResult};
use winnow::combinator::{alt, preceded, repeat, terminated, seq, dispatch, fail, opt, cut_err, trace, eof, delimited};
use winnow::Parser;
use winnow::token::{literal, take_till, take_while};
use winnow::stream::Stream;
use crate::ast::{CallArgument, FfAssignment, I64Assignment, ComponentsMode, Expr, FfExpr, I64Expr, I64Operand, Signal, Statement, Template, TemplateInstruction, AST, Function};

fn parse_prime(input: &mut &str) -> ModalResult<BigUint> {
    let (bi, ): (BigUint, ) = seq!(
        _: ("%%prime", space1),
        cut_err(digit1.parse_to()
            .context(StrContext::Label("prime"))
            .context(StrContext::Expected(StrContextValue::Description("valid prime value")))),
        _: (
            space0,
            opt(parse_eol_comment),
            cut_err(line_ending).context(StrContext::Expected(StrContextValue::CharLiteral('\n')))
        )
    ).parse_next(input)?;
    Ok(bi)
}

fn parse_signals_num(input: &mut &str) -> ModalResult<usize> {
    let (i, ): (usize, ) = seq!(
        _: ("%%signals", space1),
        cut_err(parse_usize),
        _: (
            space0,
            opt(parse_eol_comment),
            cut_err(line_ending).context(StrContext::Expected(StrContextValue::CharLiteral('\n')))
        )
    ).parse_next(input)?;
    Ok(i)
}

fn parse_components_heap(input: &mut &str) -> ModalResult<usize> {
    let (i, ): (usize, ) = seq!(
        _: ("%%components_heap", space1),
        cut_err(parse_usize),
        _: (
            space0,
            opt(parse_eol_comment),
            cut_err(line_ending).context(StrContext::Expected(StrContextValue::CharLiteral('\n')))
        )
    ).parse_next(input)?;
    Ok(i)
}

#[derive(Debug)]
enum Error {
    NotValidComponentsMode,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NotValidComponentsMode => write!(f, "Invalid components mode"),
        }
    }
}

impl std::error::Error for Error {}

fn parse_components_creation_mode(input: &mut &str) -> ModalResult<ComponentsMode> {
    let (s, ): (ComponentsMode, ) = seq!(
        _: ("%%components", space1),
        cut_err(
            alpha1.try_map(|s| match s {
                "implicit" => Ok(ComponentsMode::Implicit),
                "explicit" => Ok(ComponentsMode::Explicit),
                _ => Err(Error::NotValidComponentsMode),
            })
            .context(StrContext::Label("components mode"))
        ),
        _: (
            space0,
            opt(parse_eol_comment),
            cut_err(line_ending).context(StrContext::Expected(StrContextValue::CharLiteral('\n')))
        )
    ).parse_next(input)?;
    Ok(s)
}

fn parse_start_template(input: &mut &str) -> ModalResult<String> {
    let (s, ): (&str, ) = seq!(
        _: ("%%start", space1),
        cut_err(parse_variable_name),
        _: (
            space0,
            opt(parse_eol_comment),
            cut_err(line_ending).context(StrContext::Expected(StrContextValue::CharLiteral('\n')))
        )
    ).parse_next(input)?;
    Ok(s.to_owned())
}

fn parse_witness_list(input: &mut &str) -> ModalResult<Vec<usize>> {
    let (s, ): (Vec<usize>, ) = seq!(
        _: ("%%witness", space1),
        cut_err(parse_vec_usize),
        _: (
            space0,
            opt(parse_eol_comment),
            cut_err(line_ending).context(StrContext::Expected(StrContextValue::CharLiteral('\n')))
        )
    ).parse_next(input)?;
    Ok(s)
}

fn parse_i64_literal(input: &mut &str) -> ModalResult<i64> {
    preceded(
        literal("i64."),
        cut_err(digit1.parse_to()
            .context(StrContext::Label("i64 literal")))
            .context(StrContext::Expected(StrContextValue::Description("valid i64 value"))),)
        .parse_next(input)
}

fn parse_ff_literal(input: &mut &str) -> ModalResult<BigUint> {
    preceded(
        literal("ff."),
        cut_err(digit1.parse_to()
            .context(StrContext::Label("ff literal")))
            .context(StrContext::Expected(StrContextValue::Description("valid ff value"))),)
        .parse_next(input)
}

fn parse_i64_operand(input: &mut &str) -> ModalResult<I64Operand> {
    if let Some(x) = opt(parse_i64_literal).parse_next(input)? {
        Ok(I64Operand::Literal(x))
    } else if let Some(x) = opt(parse_variable_name).parse_next(input)? {
        Ok(I64Operand::Variable(x.to_string()))
    } else {
        fail.parse_next(input)?
    }
}

fn parse_ff_expr(input: &mut &str) -> ModalResult<FfExpr> {
    if let Some(x) = opt(parse_ff_literal).parse_next(input)? {
        Ok(FfExpr::Literal(x))
    } else if let Some(x) = opt(parse_variable_name).parse_next(input)? {
        Ok(FfExpr::Variable(x.to_string()))
    } else {
        fail.parse_next(input)?
    }
}

fn parse_call_argument(input: &mut &str) -> ModalResult<CallArgument> {
    alt((
        // Try to parse memory operations first (more specific)
        dispatch! {parse_operator_name;
            "i64.memory" => delimited(
                literal("("),
                (parse_i64_operand, preceded(literal(","), parse_i64_operand)),
                literal(")")
            ).map(|(addr, size)| CallArgument::I64Memory { addr, size }),
            "ff.memory" => delimited(
                literal("("),
                (parse_i64_operand, preceded(literal(","), parse_i64_operand)),
                literal(")")
            ).map(|(addr, size)| CallArgument::FfMemory { addr, size }),
            _ => fail::<_, CallArgument, _>,
        },
        // Try to parse literals
        parse_i64_literal.map(CallArgument::I64Literal),
        parse_ff_literal.map(CallArgument::FfLiteral),
        // Try to parse variable names
        parse_variable_name.map(|name| CallArgument::Variable(name.to_string())),
    )).parse_next(input)
}

fn parse_variable_name<'s>(input: &mut &'s str) -> ModalResult<&'s str> {
    trace(
        "parse_variable_name",
        |i: &mut _| {
            let var_name = take_while(1..,
                       |c: char| c.is_alphabetic() || c.is_numeric() || c == '_')
                .verify(|v: &str| {
                    let c = v.chars().next().unwrap();
                    c.is_alphabetic() || c == '_'
                })
                .parse_next(i)?;
            Ok(var_name)
        }
    ).parse_next(input)
}

fn parse_operator_name<'s>(input: &mut &'s str) -> ModalResult<&'s str> {
    let var_name = take_while(1..,
               |c: char| c.is_alphabetic() || c.is_numeric() || c == '_' || c == '.')
        .parse_next(input)?;
    Ok(var_name)
}

fn parse_line_end(i: &mut &str) -> ModalResult<()> {
    alt((winnow::ascii::crlf, literal("\n"), eof)).void().parse_next(i)
}

fn parse_empty_line(input: &mut &str) -> ModalResult<()> {
    trace(
        "parse_empty_line",
        |i: &mut _| {
            (
                space0,
                opt(parse_eol_comment),
                parse_line_end
            ).map(|_| ()).parse_next(i)
        }
    ).parse_next(input)
}

fn parse_instruction_line(input: &mut &str) -> ModalResult<Option<TemplateInstruction>> {
    trace(
        "parse_instruction_line",
        |i: &mut _| {
            alt((
                parse_statement.map(|x| Some(TemplateInstruction::Statement(x))),
                parse_assignment.map(|x| Some(TemplateInstruction::FfAssignment(x))),
                parse_i64_assignment.map(|x| Some(TemplateInstruction::I64Assignment(x))),
                parse_empty_line.map(|_| None),
                // cut_err(fail.context(StrContext::Label("line"))),
            )).parse_next(i)
        }
    ).parse_next(input)
}

fn parse_template_body(input: &mut &str) -> ModalResult<Vec<TemplateInstruction>> {
    let mut instructions = Vec::new();
    'outer: while !input.is_empty() {
        let inst = parse_instruction_line(input);
        match inst {
            Ok(Some(inst)) => { instructions.push(inst); }
            Ok(None) => (),
            Err(err) => {
                match err {
                    ErrMode::Cut(_) => { return Err(err); }
                    _ => { break 'outer; }
                }
            }
        }
    }
    if instructions.is_empty() {
        return Err(ErrMode::Cut(ContextError::from_input(input)));
    }
    Ok(instructions)
}

fn parse_i64_assignment(input: &mut &str) -> ModalResult<I64Assignment> {
    let (var_name, _, _, _, expr, _, _, _) = seq!(
        parse_variable_name,
        space0, literal("="), space0,
        cut_err(parse_i64_expression
            .context(StrContext::Label("expression"))
            .context(StrContext::Expected(StrContextValue::Description("valid i64 expression")))),
        space0,
        opt(parse_eol_comment),
        parse_line_end)
        .parse_next(input)?;
    Ok(I64Assignment {
        dest: var_name.to_string(),
        value: expr,
    })
}

fn parse_assignment_variable(input: &mut &str) -> ModalResult<Statement> {
    let (lhv, _, _, _, rhv, _, _, _) = seq!(
        parse_variable_name,
        space0,
        literal("="),
        space0,
        parse_variable_name,
        space0,
        opt(parse_eol_comment),
        parse_line_end
    ).parse_next(input)?;
    Ok(Statement::Assignment {
        name: lhv.to_string(),
        value: Expr::Variable(rhv.to_string()),
    })
}

fn parse_assignment(input: &mut &str) -> ModalResult<FfAssignment> {
    let (var_name, _, _, _, expr, _, _, _) = seq!(
        parse_variable_name,
        space0, literal("="), space0,
        cut_err(parse_ff_expression
            .context(StrContext::Label("expression"))
            .context(StrContext::Expected(StrContextValue::Description("valid ff expression")))),
        space0,
        opt(parse_eol_comment),
        parse_line_end)
        .parse_next(input)?;
    Ok(FfAssignment {
        dest: var_name.to_string(),
        value: expr,
    })
}

fn parse_block_until(input: &mut &str, terminators: &[&str]) -> ModalResult<Vec<TemplateInstruction>> {
    let mut block = Vec::new();
    
    while !input.is_empty() {
        // Check if we've reached any of the terminators
        // We need to ensure the terminator is a complete word, not just a prefix
        let found_terminator = terminators.iter().any(|&term| {
            if let Some(after_term) = input.strip_prefix(term) {
                // Check if the terminator is followed by a non-alphanumeric character
                // or is at the end of input
                after_term.is_empty() || 
                    after_term.chars().next().is_none_or(|c| !c.is_alphanumeric() && c != '_')
            } else {
                false
            }
        });
        
        if found_terminator {
            break;
        }

        let inst =
            cut_err(parse_instruction_line.context(StrContext::Label("line")))
                .parse_next(input)?;
        if let Some(inst) = inst {
             block.push(inst);
        }
    }
    
    Ok(block)
}

fn parse_branch_with_condition<F, T>(
    condition_parser: F,
    wrap_expr: fn(T) -> Expr,
) -> impl FnMut(&mut &str) -> ModalResult<Statement>
where
    F: Fn(&mut &str) -> ModalResult<T> + Copy,
{
    move |i: &mut &str| {
        // Parse the condition
        let (condition, _, _, _) = seq!(
            preceded(space1, condition_parser),
            space0,
            opt(parse_eol_comment),
            parse_line_end
        ).parse_next(i)?;

        // Parse the if block
        let if_block = parse_block_until(i, &["else", "end"])?;

        // Parse the optional else block
        let mut else_block = Vec::new();
        if i.starts_with("else") {
            // Consume the "else" keyword
            let _ = (literal("else"), space0, opt(parse_eol_comment), parse_line_end).parse_next(i)?;
            else_block = parse_block_until(i, &["end"])?;
        }

        // Consume the "end" keyword
        let _ = (literal("end"), space0, opt(parse_eol_comment), parse_line_end).parse_next(i)?;

        Ok(Statement::Branch {
            condition: wrap_expr(condition),
            if_block,
            else_block,
        })
    }
}

fn parse_statement(input: &mut &str) -> ModalResult<Statement> {
    let checkpoint = input.checkpoint();
    match parse_assignment_variable.parse_next(input) {
        Ok(assignment) => return Ok(assignment),
        Err(err) => {
            if let ErrMode::Cut(_) = err {
                return Err(err);
            }
        }
    }
    input.reset(&checkpoint);

    let s = dispatch! {parse_operator_name;
        "set_signal" => (preceded(space1, parse_i64_operand), preceded(space1, parse_ff_expr))
            .map(|(op1, op2)| Statement::SetSignal{ idx: op1, value: op2 }),
        "ff.store" => (preceded(space1, parse_i64_operand), preceded(space1, parse_ff_expr))
            .map(|(op1, op2)| Statement::FfStore{ idx: op1, value: op2 }),
        "set_cmp_input_run" => (
            preceded(space1, parse_i64_operand),
            preceded(space1, parse_i64_operand),
            preceded(space1, parse_ff_expr))
            .map(
                |(op1, op2, op3)| Statement::SetCmpSignalRun{ cmp_idx: op1, sig_idx: op2, value: op3 }),
        "set_cmp_input" => (
            preceded(space1, parse_i64_expression),
            preceded(space1, parse_i64_expression),
            preceded(space1, parse_ff_expr))
            .map(
                |(op1, op2, op3)| Statement::SetCmpInput{ cmp_idx: op1, sig_idx: op2, value: op3 }),
        "error" => preceded(space1, parse_i64_operand)
            .map(|op1| Statement::Error{ code: op1 }),
        "loop" => {
            |i: &mut &str| {
                // Parse the newline after "loop"
                let _ = (space0, opt(parse_eol_comment), parse_line_end).parse_next(i)?;

                // Parse the loop body
                let loop_body = parse_block_until(i, &["end"])?;

                // Consume the "end" keyword
                let _ = (literal("end"), space0, opt(parse_eol_comment), parse_line_end).parse_next(i)?;

                Ok(Statement::Loop(loop_body))
            }
        },
        "continue" => {
            |i: &mut &str| {
                // Parse the newline after "continue"
                let _ = (space0, opt(parse_eol_comment), parse_line_end).parse_next(i)?;

                Ok(Statement::Continue)
            }
        },
        "break" => {
            |i: &mut &str| {
                // Parse the newline after "break"
                let _ = (space0, opt(parse_eol_comment), parse_line_end).parse_next(i)?;

                Ok(Statement::Break)
            }
        },
        "i64.if" => parse_branch_with_condition(parse_i64_expression, Expr::I64),
        "ff.if" => parse_branch_with_condition(parse_ff_expression, Expr::Ff),
        "ff.mreturn" => (
            preceded(space1, parse_i64_operand),
            preceded(space1, parse_i64_operand),
            preceded(space1, parse_i64_operand))
            .map(
                |(op1, op2, op3)| Statement::FfMReturn{ dst: op1, src: op2, size: op3 }),
        "ff.mcall" => {
            |i: &mut &str| {
                // Parse the function name prefixed with $
                let _ = space1.parse_next(i)?;
                let _ = literal("$").parse_next(i)?;
                let name = parse_variable_name.parse_next(i)?;
                
                // Parse the arguments using a proper repeat parser
                let args = repeat(0.., preceded(space1, parse_call_argument)).parse_next(i)?;
                
                Ok(Statement::FfMCall {
                    name: name.to_string(),
                    args,
                })
            }
        },
        _ => fail::<_, Statement, _>,
    }
        .parse_next(input)?;

    // For set_signal, ff.store, set_cmp_input_run, error, ff.mreturn, and ff.mcall, we need to parse the line end
    match &s {
        Statement::SetSignal { .. } | Statement::FfStore { .. } | Statement::SetCmpSignalRun { .. } | Statement::Error { .. } | Statement::FfMReturn { .. } | Statement::FfMCall { .. } => {
            (space0, opt(parse_eol_comment), parse_line_end).parse_next(input)?;
        }
        _ => {}
    }

    Ok(s)
}

fn parse_ff_expression(input: &mut &str) -> ModalResult<FfExpr> {
    let result = alt((
        // Try to parse as an operator expression
        dispatch! {parse_operator_name;
            "get_signal" => preceded(space1, parse_i64_operand).map(FfExpr::GetSignal),
            "get_cmp_signal" => (preceded(space1, parse_i64_operand), preceded(space1, parse_i64_operand))
                .map(|(op1, op2)| FfExpr::GetCmpSignal { cmp_idx: op1, sig_idx: op2 }),
            "ff.mul" => (preceded(space1, parse_ff_expr), preceded(space1, parse_ff_expr))
                .map(|(op1, op2)| FfExpr::FfMul(Box::new(op1), Box::new(op2))),
            "ff.add" => (preceded(space1, parse_ff_expr), preceded(space1, parse_ff_expr))
                .map(|(op1, op2)| FfExpr::FfAdd(Box::new(op1), Box::new(op2))),
            "ff.neq" => (preceded(space1, parse_ff_expr), preceded(space1, parse_ff_expr))
                .map(|(op1, op2)| FfExpr::FfNeq(Box::new(op1), Box::new(op2))),
            "ff.div" => (preceded(space1, parse_ff_expr), preceded(space1, parse_ff_expr))
                .map(|(op1, op2)| FfExpr::FfDiv(Box::new(op1), Box::new(op2))),
            "ff.sub" => (preceded(space1, parse_ff_expr), preceded(space1, parse_ff_expr))
                .map(|(op1, op2)| FfExpr::FfSub(Box::new(op1), Box::new(op2))),
            "ff.eq" => (preceded(space1, parse_ff_expr), preceded(space1, parse_ff_expr))
                .map(|(op1, op2)| FfExpr::FfEq(Box::new(op1), Box::new(op2))),
            "ff.eqz" => (preceded(space1, parse_ff_expr))
                .map(|op1| FfExpr::FfEqz(Box::new(op1))),
            "ff.load" => preceded(space1, parse_i64_operand).map(FfExpr::Load),
            "ff.lt" => (preceded(space1, parse_ff_expr), preceded(space1, parse_ff_expr))
                .map(|(op1, op2)| FfExpr::Lt(Box::new(op1), Box::new(op2))),
            _ => fail::<_, FfExpr, _>,
        },
        // Try to parse as a literal
        parse_ff_literal.map(FfExpr::Literal),
        // Try to parse as a variable
        parse_variable_name.map(|name| FfExpr::Variable(name.to_string())),
    )).parse_next(input)?;

    Ok(result)
}

fn parse_i64_expression(input: &mut &str) -> ModalResult<I64Expr> {
    let result = alt((
        // Try to parse as an operator expression
        dispatch! {parse_operator_name;
            "i64.add" => (preceded(space1, parse_i64_expression), preceded(space1, parse_i64_expression))
                .map(|(op1, op2)| I64Expr::Add(Box::new(op1), Box::new(op2))),
            "i64.sub" => (preceded(space1, parse_i64_expression), preceded(space1, parse_i64_expression))
                .map(|(op1, op2)| I64Expr::Sub(Box::new(op1), Box::new(op2))),
            "i64.mul" => (preceded(space1, parse_i64_expression), preceded(space1, parse_i64_expression))
                .map(|(op1, op2)| I64Expr::Mul(Box::new(op1), Box::new(op2))),
            "i64.le" => (preceded(space1, parse_i64_expression), preceded(space1, parse_i64_expression))
                .map(|(op1, op2)| I64Expr::Lte(Box::new(op1), Box::new(op2))),
            "i64.load" => preceded(space1, parse_i64_operand)
                .map(I64Expr::Load),
            "i64.wrap_ff" => preceded(space1, parse_ff_expr)
                .map(|expr| I64Expr::Wrap(Box::new(expr))),
            _ => fail::<_, I64Expr, _>,
        },
        // Try to parse as a literal
        parse_i64_literal.map(I64Expr::Literal),
        // Try to parse as a variable
        parse_variable_name.map(|name| I64Expr::Variable(name.to_string())),
    )).parse_next(input)?;

    Ok(result)
}


fn parse_eol_comment(i: &mut &str) -> ModalResult<()> {
    (";;", take_till(0.., ['\n', '\r']))
        .void() // Output is thrown away.
        .parse_next(i)
}

fn parse_usize(i: &mut &str) -> ModalResult<usize> {
    digit1.parse_to().context(StrContext::Label("usize")).parse_next(i)
}

fn parse_ff_signal(i: &mut &str) -> ModalResult<Signal> {

    let (_, _, dims_num, sp) = (literal("ff"), space1, parse_usize, space0).parse_next(i)?;

    if dims_num == 0 {
        return Ok(Signal::Ff(vec![]));
    }

    assert!(!sp.is_empty());

    let dims: Vec<usize> = repeat(
        dims_num..=dims_num, terminated(parse_usize, space0))
        .parse_next(i)?;

    assert_eq!(dims.len(), dims_num);
    Ok(Signal::Ff(dims))
}

fn parse_opt_usize(i: &mut &str) -> ModalResult<Option<usize>> {
    let (sign, digs) = (opt('-'), digit1).parse_next(i)?;
    match sign {
        Some(_) => {
            if digs == "1" {
                Ok(None)
            } else {
                Err(ErrMode::Cut(ContextError::from_input(i)))
            }
        }
        None => {
            let val = digs.parse()
                .map_err(|_| ErrMode::Cut(ContextError::from_input(i)))?;
            Ok(Some(val))
        }
    }
}

fn parse_vec_opt_usize(input: &mut &str) -> ModalResult<Vec<Option<usize>>> {
    repeat(0.., terminated(parse_opt_usize, space0))
        .parse_next(input)
}

fn parse_vec_usize(input: &mut &str) -> ModalResult<Vec<usize>> {
    repeat(0.., terminated(parse_usize, space0)).parse_next(input)
}

fn parse_bus_signal(i: &mut &str) -> ModalResult<Signal> {
    let (_, bus_id, _, dims_num, sp) =
        (literal("bus_"), digit1, space1, parse_usize, space0).parse_next(i)?;
    let bus_name = format!("bus_{}", bus_id);

    if dims_num == 0 {
        return Ok(Signal::Bus(bus_name, vec![]));
    }

    assert!(!sp.is_empty());

    let dims: Vec<usize> = repeat(
        dims_num..=dims_num, terminated(parse_usize, space0))
        .parse_next(i)?;

    assert_eq!(dims.len(), dims_num);
    Ok(Signal::Bus(bus_name, dims))
}

fn parse_signals(i: &mut &str) -> ModalResult<Vec<Signal>> {
    delimited(
        terminated('[', space0),
        repeat(0.., alt((parse_ff_signal, parse_bus_signal))),
        ']').parse_next(i)
}

fn parse_template(i: &mut &str) -> ModalResult<Template> {
    seq!{Template{
        _: (literal("%%template"), space1),
        name: parse_variable_name.map(|s: &str| s.to_owned()),
        _: space1,
        outputs: parse_signals,
        _: space0,
        inputs: parse_signals,
        _: (space0, '[', space0),
        signals_num: parse_usize,
        _: (space0, ']', space0, '[', space0),
        components: parse_vec_opt_usize,
        _: (space0, ']', space0, line_ending),
        body: parse_template_body,
    }}.parse_next(i)
}

fn parse_function_output_type(i: &mut &str) -> ModalResult<()> {
    // Parse function output type: [], [ff], or [i64]
    delimited(
        terminated('[', space0),
        opt(alt((literal("ff"), literal("i64")))),
        terminated(']', space0)
    ).void().parse_next(i)
}

fn parse_function_param(i: &mut &str) -> ModalResult<()> {
    // Parse a single function parameter: type (ff or i64) followed by dimensions
    // For now, we just consume and ignore the parameter definition
    let _ = alt((literal("ff"), literal("i64"))).parse_next(i)?;
    let _ = space1.parse_next(i)?;
    
    // Parse number of dimensions
    let dims_num: usize = parse_usize.parse_next(i)?;
    let _ = space0.parse_next(i)?;
    
    // Parse dimension sizes
    for _ in 0..dims_num {
        let _ = parse_usize.parse_next(i)?;
        let _ = space0.parse_next(i)?;
    }
    
    Ok(())
}

fn parse_function_params(i: &mut &str) -> ModalResult<()> {
    // Parse the function parameters list
    delimited(
        terminated('[', space0),
        repeat::<_, _, (), _, _>(0.., parse_function_param),
        terminated(']', space0)
    ).void().parse_next(i)
}

fn parse_function(i: &mut &str) -> ModalResult<Function> {
    seq!{Function{
        _: (literal("%%function"), space1),
        name: parse_variable_name.map(|s: &str| s.to_owned()),
        _: space1,
        // Parse output type
        _: parse_function_output_type,
        // Parse input parameters
        _: parse_function_params,
        _: line_ending,
        body: parse_template_body,
    }}.parse_next(i)
}

fn parse_ast(i: &mut &str) -> ModalResult<AST> {
    seq!{AST{
        _: repeat::<_, _, (), _, _>(0.., parse_empty_line),
        prime: parse_prime
            .context(StrContext::Expected(StrContextValue::StringLiteral("%%prime"))),
        _: repeat::<_, _, (), _, _>(0.., parse_empty_line),
        signals: parse_signals_num
            .context(StrContext::Expected(StrContextValue::StringLiteral("%%signals"))),
        _: repeat::<_, _, (), _, _>(0.., parse_empty_line),
        components_heap: parse_components_heap
            .context(StrContext::Expected(StrContextValue::StringLiteral("%%components_heap"))),
        _: repeat::<_, _, (), _, _>(0.., parse_empty_line),
        start: parse_start_template
            .context(StrContext::Expected(StrContextValue::StringLiteral("%%start"))),
        _: repeat::<_, _, (), _, _>(0.., parse_empty_line),
        components_mode: parse_components_creation_mode
            .context(StrContext::Expected(StrContextValue::StringLiteral("%%components"))),
        _: repeat::<_, _, (), _, _>(0.., parse_empty_line),
        witness: parse_witness_list
            .context(StrContext::Expected(StrContextValue::StringLiteral("%%witness"))),
        _: repeat::<_, _, (), _, _>(0.., parse_empty_line),
        functions: repeat(0.., parse_function),
        _: repeat::<_, _, (), _, _>(0.., parse_empty_line),
        templates: repeat(1.., parse_template),
    }}.parse_next(i)
}

pub fn parse(i: &str) -> Result<AST, ParseError<&str, ContextError>> {
    parse_ast.parse(i)
}

#[cfg(test)]
mod tests {
    use num_traits::Num;
    use num_bigint::BigUint;
    use winnow::{Parser};
    use winnow::stream::{Offset, Stream};
    use super::*;

    #[test]
    fn test_parse_block_until() {
        // Test 1: Basic block parsing until "end"
        let mut input = r#"x_1 = get_signal i64.1
x_2 = ff.mul x_0 x_1
set_signal i64.0 x_2
end
remaining"#;
        let block = parse_block_until(&mut input, &["end"]).unwrap();
        assert_eq!(block.len(), 3);
        assert_eq!(input, "end\nremaining");

        // Test 2: Block parsing with multiple terminators
        let mut input = r#"x_1 = ff.1
x_2 = x_1
else
x_3 = ff.0
end"#;
        let block = parse_block_until(&mut input, &["else", "end"]).unwrap();
        assert_eq!(block.len(), 2);
        assert_eq!(input, "else\nx_3 = ff.0\nend");

        // Test 3: Empty block
        let mut input = "end";
        let block = parse_block_until(&mut input, &["end"]).unwrap();
        assert_eq!(block.len(), 0);
        assert_eq!(input, "end");

        // Test 4: Block with comments and empty lines
        let mut input = r#"x_1 = ff.1 ;; comment

;; another comment
x_2 = x_1
end"#;
        let block = parse_block_until(&mut input, &["end"]).unwrap();
        assert_eq!(block.len(), 2);
        assert_eq!(input, "end");

        // Test 5: Nested control structures
        let mut input = r#"x_1 = ff.1
ff.if x_1
x_2 = ff.2
end
x_3 = ff.3
end"#;
        let block = parse_block_until(&mut input, &["end"]).unwrap();
        assert_eq!(block.len(), 3); // assignment, if statement, assignment
        assert_eq!(input, "end");

        // Test 6: Error propagation - invalid statement
        let mut input = r#"x_1 = ff.1
invalid statement here
x_2 = ff.2
end"#;
        let want_err_offset = input.find("invalid statement here").unwrap();
        let start = input.checkpoint();
        let result = parse_block_until(&mut input, &["end"]);
        assert_eq!(input.offset_from(&start), want_err_offset);
        assert_eq!(
            result.unwrap_err().into_inner().unwrap().to_string(),
            "invalid line".to_string());

        // Test 7: Multiple terminators at different positions
        let mut input = r#"x_1 = ff.1
x_2 = ff.2
else
x_3 = ff.3
end"#;
        // Should stop at first terminator found
        let block = parse_block_until(&mut input, &["end", "else"]).unwrap();
        assert_eq!(block.len(), 2);
        assert_eq!(input, "else\nx_3 = ff.3\nend");

        // Test 8: Ignore words that start with "end" but are not the terminator
        let mut input = r#"x_1 = get_signal i64.1
x_2 = ff.mul x_0 x_1
set_signal i64.0 x_2
set;; comment line
remaining"#;
        let block = parse_block_until(&mut input, &["set"]).unwrap();
        println!("{:?}", block);
        println!("{:?}", input);
        assert_eq!(block.len(), 3);
        assert_eq!(input, "set;; comment line\nremaining");
    }

    #[test]
    fn test_parse_branch() {
        let input = r#"ff.if x_1
;; store bucket. Line 30
;; getting src
;; compute bucket
;; load bucket
;; end of load bucket
x_2 = get_signal i64.1
;; OP(DIV)
x_3 = ff.div ff.1 x_2
;; end of compute bucket
;; getting dest
set_signal i64.2 x_3
;; end of store bucket
else
;; store bucket. Line 30
;; getting src
;; getting dest
set_signal i64.2 ff.0
;; end of store bucket
end
"#;

        let statement = parse_statement.parse(input).unwrap();

        let want = Statement::Branch {
            condition: Expr::Ff(ff("x_1")),
            if_block: vec![
                assign("x_2", &get_signal("1")),
                assign("x_3", &ff_div("1", "x_2")),
                set_signal("2", "x_3"),
            ],
            else_block: vec![
                set_signal("2", "0"),
            ],
        };
        assert_eq!(statement, want);
    }

    #[test]
    fn test_parse_branch2() {
        let input = r#"ff.if get_signal x_1
x_2 = get_signal i64.1
x_3 = ff.div ff.1 x_2
set_signal i64.2 x_3
else
set_signal i64.2 ff.0
end
"#;

        let statement = parse_statement.parse(input).unwrap();

        let want = Statement::Branch {
            condition: Expr::Ff(get_signal("x_1")),
            if_block: vec![
                assign("x_2", &get_signal("1")),
                assign("x_3", &ff_div("1", "x_2")),
                set_signal("2", "x_3"),
            ],
            else_block: vec![
                set_signal("2", "0"),
            ],
        };
        assert_eq!(statement, want);
    }

    #[test]
    fn test_parse_branch_i64() {
        let input = r#"i64.if x_1
x_2 = get_signal i64.1
x_3 = ff.div ff.1 x_2
set_signal i64.2 x_3
else
set_signal i64.2 ff.0
end
"#;

        let statement = parse_statement.parse(input).unwrap();

        let want = Statement::Branch {
            condition: Expr::I64(i64_expr("x_1")),
            if_block: vec![
                assign("x_2", &get_signal("1")),
                assign("x_3", &ff_div("1", "x_2")),
                set_signal("2", "x_3"),
            ],
            else_block: vec![
                set_signal("2", "0"),
            ],
        };
        assert_eq!(statement, want);
    }

    #[test]
    fn test_parse_variable_name() {
        let mut input = "my_variable_123 = get_signal";
        let var_name = parse_variable_name.parse_next(&mut input).unwrap();
        assert_eq!(var_name, "my_variable_123");
        assert_eq!(input, " = get_signal");

        let mut input = "3my_variable_123 = get_signal";
        assert!(parse_variable_name.parse_next(&mut input).is_err());
    }

    #[test]
    fn test_parse_operator_name() {
        let mut input = "my_variable_123 = get_signal";
        let op_name = parse_operator_name.parse_next(&mut input).unwrap();
        assert_eq!(op_name, "my_variable_123");
        assert_eq!(input, " = get_signal");

        let mut input = "ff.add = get_signal";
        let op_name = parse_operator_name.parse_next(&mut input).unwrap();
        assert_eq!(op_name, "ff.add");
        assert_eq!(input, " = get_signal");
    }

    #[test]
    fn test_i64_operand() {
        let mut input = "my_variable_123 = get_signal";
        let want = I64Operand::Variable("my_variable_123".to_string());
        let op = parse_i64_operand.parse_next(&mut input).unwrap();
        assert_eq!(op, want);

        let mut input = "i64.4 = get_signal";
        let want = I64Operand::Literal(4);
        let op = parse_i64_operand.parse_next(&mut input).unwrap();
        assert_eq!(op, want);

        let input = "i64.93841982938198593829123 = get_signal";
        let want_err = "\
i64.93841982938198593829123 = get_signal
    ^
invalid i64 literal
expected valid i64 value";
        let op = parse_i64_operand.parse(input)
            .unwrap_err().to_string();

        assert_eq!(op, want_err);
    }

    #[test]
    fn test_parse_expression() {
        let input = "get_signal x_50";
        let want = get_signal("x_50");
        let op = parse_ff_expression.parse(input).unwrap();
        assert_eq!(op, want);

        let input = "get_signal i64.2";
        let want = get_signal("2");
        let op = parse_ff_expression.parse(input).unwrap();
        assert_eq!(op, want);

        let input = "ff.div ff.2 v_3";
        let want = ff_div("2", "v_3");
        let op = parse_ff_expression.parse(input).unwrap();
        assert_eq!(op, want);

        let input = "v_3";
        let want = ff("v_3");
        let op = parse_ff_expression.parse(input).unwrap();
        assert_eq!(op, want);

        let input = "ff.5";
        let want = ff("5");
        let op = parse_ff_expression.parse(input).unwrap();
        assert_eq!(op, want);

        // test ff.load with literal operand
        let input = "ff.load i64.15";
        let want = FfExpr::Load(I64Operand::Literal(15));
        let op = parse_ff_expression.parse(input).unwrap();
        assert_eq!(op, want);

        // test ff.load with variable operand
        let input = "ff.load x_45";
        let want = FfExpr::Load(I64Operand::Variable("x_45".to_string()));
        let op = parse_ff_expression.parse(input).unwrap();
        assert_eq!(op, want);

        // test ff.lt expression
        let input = "ff.lt x_3 ff.2";
        let want = FfExpr::Lt(
            Box::new(FfExpr::Variable("x_3".to_string())),
            Box::new(FfExpr::Literal(BigUint::from(2u32)))
        );
        let op = parse_ff_expression.parse(input).unwrap();
        assert_eq!(op, want);
    }

    #[test]
    fn test_assignment() {
        let input = "x_4 = get_signal i64.2";
        let want = FfAssignment {
            dest: "x_4".to_string(),
            value: get_signal("2"),
        };
        let a = parse_assignment.parse(input).unwrap();
        assert_eq!(a, want);

        let mut input = "x_4 = get_signal i64.2 2";
        assert!(parse_assignment.parse_next(&mut input).is_err());

        let mut input = "x_4 = get_signal i64.2\nxxx";
        let want = FfAssignment {
            dest: "x_4".to_string(),
            value: get_signal("2"),
        };
        let a = parse_assignment.parse_next(&mut input).unwrap();
        assert_eq!(a, want);
        assert_eq!(input, "xxx");

        let mut input = "x_4 = get_signal i64.2;;\nxxx";
        let want = FfAssignment {
            dest: "x_4".to_string(),
            value: get_signal("2"),
        };
        let a = parse_assignment.parse_next(&mut input).unwrap();
        assert_eq!(a, want);
        assert_eq!(input, "xxx");

        let mut input = "x_4 = get_signal i64.2 ;; comment \nxxx";
        let want = FfAssignment {
            dest: "x_4".to_string(),
            value: get_signal("2"),
        };
        let a = parse_assignment.parse_next(&mut input).unwrap();
        assert_eq!(a, want);
        assert_eq!(input, "xxx");

        // test consumes a new line
        let mut input = "x_1 = get_signal i64.2
;; OP(MUL)
x_2 = ff.mul x_0 x_1";
        let want = FfAssignment {
            dest: "x_1".to_string(),
            value: get_signal("2"),
        };
        let a = parse_assignment.parse_next(&mut input).unwrap();
        assert_eq!(a, want);
        let want_left = ";; OP(MUL)
x_2 = ff.mul x_0 x_1";
        assert_eq!(input, want_left);
    }

    #[test]
    fn test_statement() {
        let input = "set_signal i64.0 x_3";
        let want = Statement::SetSignal {
            idx: I64Operand::Literal(0),
            value: ff("x_3"),
        };
        let a = parse_statement.parse(input).unwrap();
        assert_eq!(a, want);

        let mut input = "set_signal i64.0 x_3 ;; comment
x";
        let want = Statement::SetSignal {
            idx: I64Operand::Literal(0),
            value: ff("x_3"),
        };
        let a = parse_statement.parse_next(&mut input).unwrap();
        assert_eq!(a, want);
        assert_eq!("x", input);

        // test spaces in the end of the line
        let mut input = "set_signal i64.0 x_3
x";
        let want = Statement::SetSignal {
            idx: I64Operand::Literal(0),
            value: ff("x_3"),
        };
        let a = parse_statement.parse_next(&mut input).unwrap();
        assert_eq!(a, want);
        assert_eq!("x", input);

        // test ff.store with variable operands
        let input = "ff.store x_29 x_31";
        let want = Statement::FfStore {
            idx: I64Operand::Variable("x_29".to_string()),
            value: ff("x_31"),
        };
        let a = parse_statement.parse(input).unwrap();
        assert_eq!(a, want);

        // test ff.store with literal operands
        let input = "ff.store i64.6 ff.7511745149465107256748700652201246547602992235352608707588321460060273774987";
        let want = Statement::FfStore {
            idx: I64Operand::Literal(6),
            value: ff("7511745149465107256748700652201246547602992235352608707588321460060273774987"),
        };
        let a = parse_statement.parse(input).unwrap();
        assert_eq!(a, want);

        // test ff.store with comment
        let mut input = "ff.store i64.10 ff.123 ;; storing value
x";
        let want = Statement::FfStore {
            idx: I64Operand::Literal(10),
            value: ff("123"),
        };
        let a = parse_statement.parse_next(&mut input).unwrap();
        assert_eq!(a, want);
        assert_eq!("x", input);
    }

    #[test]
    fn test_parse_i64_literal_error() {
        let input = "i64.93841982938198593829123 = get_signal";
        let op = parse_i64_literal.parse(input).unwrap_err();
        let want_err = r#"i64.93841982938198593829123 = get_signal
    ^
invalid i64 literal
expected valid i64 value"#;
        assert_eq!(want_err, op.to_string().as_str());
    }

    #[test]
    fn test_parse_eol_comment() {
        let mut input = ";; xxx";
        assert!(parse_eol_comment.parse_next(&mut input).is_ok());
        assert_eq!(input, "");

        let mut input = ";; xxx
";
        assert!(parse_eol_comment.parse_next(&mut input).is_ok());
        assert_eq!(input, "\n");
    }

    #[test]
    fn test_ff_signal() {
        let mut input = "ff 0  ff 0 ] [ ff 0 ] [3] [ ]";
        let want = Signal::Ff(vec![]);
        let ff_signal = parse_ff_signal.parse_next(&mut input).unwrap();
        assert_eq!(want, ff_signal);
        assert_eq!(input, "ff 0 ] [ ff 0 ] [3] [ ]");

        let mut input = "ff 2 4 3  ff 0 ] [ ff 0 ] [3] [ ]";
        let want = Signal::Ff(vec![4, 3]);
        let ff_signal = parse_ff_signal.parse_next(&mut input).unwrap();
        assert_eq!(want, ff_signal);
        assert_eq!(input, "ff 0 ] [ ff 0 ] [3] [ ]");
    }

    #[test]
    fn test_bus_signal() {
        let mut input = "bus_2 0 ff 0 ] [ ff 0 ] [3] [ ]";
        let want = Signal::Bus("bus_2".to_string(), vec![]);
        let bus_signal = parse_bus_signal.parse_next(&mut input).unwrap();
        assert_eq!(want, bus_signal);
        assert_eq!(input, "ff 0 ] [ ff 0 ] [3] [ ]");

        let mut input = "bus_2 2 4 3  ff 0 ] [ ff 0 ] [3] [ ]";
        let want = Signal::Bus("bus_2".to_string(), vec![4, 3]);
        let bus_signal = parse_bus_signal.parse_next(&mut input).unwrap();
        assert_eq!(want, bus_signal);
        assert_eq!(input, "ff 0 ] [ ff 0 ] [3] [ ]");
    }

    #[test]
    fn test_template_body_i64_assignment() {
        let mut input = "x_1 = get_signal i64.2
;; OP(MUL)
x_2 = i64.1
x_3 = i64.add x_2 i64.1
x_4 = i64.sub x_2 x_3";
        let want = vec![
            assign("x_1", &get_signal("2")),
            assigni("x_2", &i64_expr("1")),
            assigni("x_3", &i64_add("x_2", "1")),
            assigni("x_4", &i64_sub("x_2", "x_3")),
        ];
        let template_body = parse_template_body.parse_next(&mut input).unwrap();
        assert_eq!(want, template_body);
        assert_eq!(input, "");
    }

    #[test]
    fn test_template_body_loop() {
        // the test template is last in the input
        let mut input = "loop
i64.if x_7
x_6 = get_signal i64.3
set_cmp_input i64.0 i64.1 x_6
x_7 = i64.sub x_7 i64.1
x_8 = i64.add x_8 i64.1
x_9 = i64.add x_9 i64.1
continue
end
break
end";
        let want = vec![
            TemplateInstruction::Statement(Statement::Loop(vec![
                TemplateInstruction::Statement(Statement::Branch {
                    condition: Expr::I64(i64_expr("x_7")),
                    if_block: vec![
                        assign("x_6", &get_signal("3")),
                        set_cmp_input("0", "1", "x_6"),
                        assigni("x_7", &i64_sub("x_7", "1")),
                        assigni("x_8", &i64_add("x_8", "1")),
                        assigni("x_9", &i64_add("x_9", "1")),
                        TemplateInstruction::Statement(Statement::Continue),
                    ],
                    else_block: vec![],
                }),
                TemplateInstruction::Statement(Statement::Break),
            ])),
        ];
        let template_body = parse_template_body.parse_next(&mut input).unwrap();
        assert_eq!(want, template_body);
        assert_eq!(input, "");
    }

    #[test]
    fn test_template_body() {
        // the test template is last in the input
        let mut input = "x_1 = get_signal i64.2
;; OP(MUL)
x_2 = ff.mul x_0 x_1
;; end of compute bucket
;; OP(ADD)
x_3 = ff.add x_2 ff.2
;; end of compute bucket
;; getting dest
set_signal i64.0 x_3";
        let want = vec![
            assign("x_1", &get_signal("2")),
            assign("x_2", &ff_mul("x_0", "x_1")),
            assign("x_3", &ff_add("x_2", "2")),
            set_signal("0", "x_3"),
        ];
        let template_body = parse_template_body.parse_next(&mut input).unwrap();
        assert_eq!(want, template_body);
        assert_eq!(input, "");

        // the test template is NOT last in the input
        let mut input = "x_2 = ff.mul x_0 x_1
;; end of compute bucket
;; getting dest
set_signal i64.0 x_2
;; end of store bucket


%%template Tmpl2_1 [ ff 1 3] [ ff 0 ] [4] [ ]
;; store bucket. Line 26";
        let want = vec![
            assign("x_2", &ff_mul("x_0", "x_1")),
            set_signal("0", "x_2"),
        ];
        let template_body = parse_template_body.parse_next(&mut input).unwrap();
        assert_eq!(want, template_body);
        assert_eq!(input, "%%template Tmpl2_1 [ ff 1 3] [ ff 0 ] [4] [ ]
;; store bucket. Line 26");
    }

    #[test]
    fn test_parse_function() {
        let input = "%%function f1_0 [] [ i64 0 i64 0 ff 0  ff 2 2 2]
;; compute bucket
;; load bucket
x_2 = ff.mul x_0 x_1
;; end of compute bucket
;; OP(ADD)
x_3 = ff.add x_2 ff.2
;; end of store bucket";
        let want = Function {
            name: "f1_0".to_string(),
            body: vec![
                assign("x_2", &ff_mul("x_0", "x_1")),
                assign("x_3", &ff_add("x_2", "2")),
            ],
        };
        let tmpl_header = consume_parse_result(parse_function.parse(input));
        assert_eq!(want, tmpl_header);
    }

    #[ignore]
    #[test]
    fn test_parse_function_2() {
        let input = "%%function f1_0 [] [ i64 0 i64 0 ff 0  ff 2 2 2]
x_0 = i64.26
x_1 = i64.load i64.0
x_2 = i64.load i64.1
;; store bucket in function
;;line 6
;; getting dest
ff.store i64.5 ff.0
;; end of store bucket
;; store bucket in function
;;line 7
;; getting dest
ff.store i64.6 ff.0
;; end of store bucket
;; loop bucket. Line 7
loop
;; compute bucket
;; load bucket
x_3 = ff.load i64.6
;; end of load bucket
;; OP(LESSER)
x_4 = ff.lt x_3 ff.2
;; end of compute bucket
ff.if x_4
;; store bucket in function
;;line 8
;; getting dest
ff.store i64.7 ff.0
;; end of store bucket
;; loop bucket. Line 8
loop
;; compute bucket
;; load bucket
x_5 = ff.load i64.7
;; end of load bucket
;; OP(LESSER)
x_6 = ff.lt x_5 ff.2
;; end of compute bucket
ff.if x_6
;; store bucket in function
;;line 9
;; compute bucket
;; load bucket
x_7 = ff.load i64.5
;; end of load bucket
;; load bucket
;; compute bucket
;; compute bucket
;; compute bucket
;; compute bucket
;; load bucket
x_8 = ff.load i64.6
;; end of load bucket
;; OP(TO_ADDRESS)
x_9 = i64.wrap_ff x_8
;; end of compute bucket
;; OP(MUL_ADDRESS)
;;line 0
x_10 = i64.mul i64.2 x_9
;; end of compute bucket
;; compute bucket
;; compute bucket
;; load bucket
;;line 9
x_11 = ff.load i64.7
;; end of load bucket
;; OP(TO_ADDRESS)
x_12 = i64.wrap_ff x_11
;; end of compute bucket
;; OP(MUL_ADDRESS)
;;line 0
x_13 = i64.mul i64.1 x_12
;; end of compute bucket
;; OP(ADD_ADDRESS)
x_14 = i64.add x_10 x_13
;; end of compute bucket
;; OP(ADD_ADDRESS)
x_15 = i64.add x_14 i64.1
;; end of compute bucket
x_16 = ff.load x_15
;; end of load bucket
;; OP(ADD)
;;line 9
x_17 = ff.add x_7 x_16
;; end of compute bucket
;; getting dest
ff.store i64.5 x_17
;; end of store bucket
;; store bucket in function
;;line 8
;; compute bucket
;; load bucket
x_18 = ff.load i64.7
;; end of load bucket
;; OP(ADD)
x_19 = ff.add x_18 ff.1
;; end of compute bucket
;; getting dest
ff.store i64.7 x_19
;; end of store bucket
continue
end
end
;; end of loop bucket
;; store bucket in function
;;line 7
;; compute bucket
;; load bucket
x_20 = ff.load i64.6
;; end of load bucket
;; OP(ADD)
x_21 = ff.add x_20 ff.1
;; end of compute bucket
;; getting dest
ff.store i64.6 x_21
;; end of store bucket
continue
end
end
;; end of loop bucket
;; branch bucket
;; compute bucket
;; load bucket
;;line 12
x_22 = ff.load i64.0
;; end of load bucket
;; OP(EQ(Single(1)))
x_23 = ff.eq x_22 ff.2
;; end of compute bucket
ff.if x_23
;; store bucket in function
;;line 15
;; getting dest
ff.store i64.6 ff.2910766817845651019878574839501801340070030115151021261302834310722729507541
;; end of store bucket
;; store bucket in function
;; getting dest
ff.store i64.7 ff.5776684794125549462448597414050232243778680302179439492664047328281728356345
;; end of store bucket
;; store bucket in function
;;line 19
;; getting dest
ff.store i64.8 ff.19727366863391167538122140361473584127147630672623100827934084310230022599144
;; end of store bucket
;; store bucket in function
;; getting dest
ff.store i64.9 ff.8348174920934122550483593999453880006756108121341067172388445916328941978568
;; end of store bucket
;; store bucket in function
;;line 13
x_24 = i64.6
x_25 = i64.10
x_26 = i64.2
loop
i64.if x_26 
x_27 = ff.load x_24
ff.store x_25 x_27
x_24 = i64.add x_24 i64.1
x_25 = i64.add x_25 i64.1
x_26 = i64.sub x_26 i64.1
continue
end
break
end
;; end of store bucket
;; store bucket in function
x_28 = i64.8
x_29 = i64.12
x_30 = i64.2
loop
i64.if x_30 
x_31 = ff.load x_28
ff.store x_29 x_31
x_28 = i64.add x_28 i64.1
x_29 = i64.add x_29 i64.1
x_30 = i64.sub x_30 i64.1
continue
end
break
end
;; end of store bucket
;; return bucket
;; load bucket
x_32 = ff.load i64.10
;; end of load bucket
x_33 = i64.le i64.4 x_1
i64.if x_33
x_34 = i64.4
else
x_34 = x_1
end
ff.mreturn x_1 x_32 x_34
;;line 12
else
;; branch bucket
;; compute bucket
;; load bucket
;;line 24
x_35 = ff.load i64.0
;; end of load bucket
;; OP(EQ(Single(1)))
x_36 = ff.eq x_35 ff.3
;; end of compute bucket
ff.if x_36
;; store bucket in function
;;line 27
;; getting dest
ff.store i64.6 ff.7511745149465107256748700652201246547602992235352608707588321460060273774987
;; end of store bucket
;; store bucket in function
;; getting dest
ff.store i64.7 ff.18732019378264290557468133440468564866454307626475683536618613112504878618481
;; end of store bucket
;; store bucket in function
;; getting dest
ff.store i64.8 ff.9131299761947733513298312097611845208338517739621853568979632113419485819303
;; end of store bucket
;; store bucket in function
;;line 32
;; getting dest
ff.store i64.9 ff.10370080108974718697676803824769673834027675643658433702224577712625900127200
;; end of store bucket
;; store bucket in function
;; getting dest
ff.store i64.10 ff.20870176810702568768751421378473869562658540583882454726129544628203806653987
;; end of store bucket
;; store bucket in function
;; getting dest
ff.store i64.11 ff.10595341252162738537912664445405114076324478519622938027420701542910180337937
;; end of store bucket
;; store bucket in function
;;line 37
;; getting dest
ff.store i64.12 ff.19705173408229649878903981084052839426532978878058043055305024233888854471533
;; end of store bucket
;; store bucket in function
;; getting dest
ff.store i64.13 ff.7266061498423634438633389053804536045105766754026813321943009179476902321146
;; end of store bucket
;; store bucket in function
;; getting dest
ff.store i64.14 ff.11597556804922396090267472882856054602429588299176362916247939723151043581408
;; end of store bucket
;; store bucket in function
;;line 25
x_37 = i64.6
x_38 = i64.15
x_39 = i64.3
loop
i64.if x_39 
x_40 = ff.load x_37
ff.store x_38 x_40
x_37 = i64.add x_37 i64.1
x_38 = i64.add x_38 i64.1
x_39 = i64.sub x_39 i64.1
continue
end
break
end
;; end of store bucket
;; store bucket in function
x_41 = i64.9
x_42 = i64.18
x_43 = i64.3
loop
i64.if x_43 
x_44 = ff.load x_41
ff.store x_42 x_44
x_41 = i64.add x_41 i64.1
x_42 = i64.add x_42 i64.1
x_43 = i64.sub x_43 i64.1
continue
end
break
end
;; end of store bucket
;; store bucket in function
x_45 = i64.12
x_46 = i64.21
x_47 = i64.3
loop
i64.if x_47 
x_48 = ff.load x_45
ff.store x_46 x_48
x_45 = i64.add x_45 i64.1
x_46 = i64.add x_46 i64.1
x_47 = i64.sub x_47 i64.1
continue
end
break
end
;; end of store bucket
;; return bucket
;; load bucket
x_49 = ff.load i64.15
;; end of load bucket
x_50 = i64.le i64.9 x_1
i64.if x_50
x_51 = i64.9
else
x_51 = x_1
end
ff.mreturn x_1 x_49 x_51
;;line 24
else
;; assert bucket
;;line 44
x_52 = ff.eqz ff.0
ff.if x_52
error i64.0
end
;; end of assert bucket
;; store bucket in function
;;line 45
;; getting dest
ff.store i64.6 ff.0
;; end of store bucket
;; store bucket in function
;; load bucket
x_53 = ff.load i64.6
;; end of load bucket
;; getting dest
ff.store i64.7 x_53
;; end of store bucket
;; return bucket
;; load bucket
x_54 = ff.load i64.7
;; end of load bucket
x_55 = i64.le i64.1 x_1
i64.if x_55
x_56 = i64.1
else
x_56 = x_1
end
ff.mreturn x_1 x_54 x_56
end
;; end of branch bucket
end
;; end of branch bucket";
        // let want = Function {
        //     name: "f1_0".to_string(),
        //     body: vec![
        //         assign("x_2", &ff_mul("x_0", "x_1")),
        //         assign("x_3", &ff_add("x_2", "2")),
        //     ],
        // };
        // let tmpl_header = consume_parse_result(parse_function.parse(input));
        consume_parse_result(parse_function.parse(input));
        // assert_eq!(want, tmpl_header);
    }

    #[test]
    fn test_parse_template() {
        let input = "%%template Multiplier_0 [ ff 0  ff 0 ] [ ff 0 ] [3] [ ]
;; store bucket. Line 15
;; getting src
;; compute bucket
;; compute bucket
;; load bucket
;; end of load bucket
x_0 = get_signal i64.1
;; load bucket
;; end of load bucket
x_1 = get_signal i64.2
;; OP(MUL)
x_2 = ff.mul x_0 x_1
;; end of compute bucket
;; OP(ADD)
x_3 = ff.add x_2 ff.2
;; end of compute bucket
;; getting dest
set_signal i64.0 x_3
;; end of store bucket";
        let want = Template {
            name: "Multiplier_0".to_string(),
            outputs: vec![Signal::Ff(vec![]), Signal::Ff(vec![])],
            inputs: vec![Signal::Ff(vec![])],
            signals_num: 3,
            components: vec![],
            body: vec![
                assign("x_0", &get_signal("1")),
                assign("x_1", &get_signal("2")),
                assign("x_2", &ff_mul("x_0", "x_1")),
                assign("x_3", &ff_add("x_2", "2")),
                set_signal("0", "x_3"),
            ],
        };
        let tmpl_header = parse_template.parse(input).unwrap();
        assert_eq!(want, tmpl_header);

        let mut input = "%%template Multiplier_3 [ ] [] [3] [ 0 -1 2 2]
;; store bucket. Line 15
;; getting src
;; compute bucket
;; compute bucket
;; load bucket
;; end of load bucket
x_0 = get_signal i64.1
;; load bucket
;; end of load bucket
x_1 = get_signal i64.2
;; OP(MUL)
x_2 = ff.mul x_0 x_1
;; end of compute bucket
;; OP(ADD)
x_3 = ff.add x_2 ff.2
;; end of compute bucket
;; getting dest
set_signal i64.0 x_3
;; end of store bucket

%%template Multiplier_0 [ ff 0  ff 0 ] [ ff 0 ] [3] [ ]";
        let want = Template {
            name: "Multiplier_3".to_string(),
            outputs: vec![],
            inputs: vec![],
            signals_num: 3,
            components: vec![Some(0), None, Some(2), Some(2)],
            body: vec![
                assign("x_0", &get_signal("1")),
                assign("x_1", &get_signal("2")),
                assign("x_2", &ff_mul("x_0", "x_1")),
                assign("x_3", &ff_add("x_2", "2")),
                set_signal("0", "x_3"),
            ],
        };
        let tmpl_header = parse_template.parse_next(&mut input).unwrap();
        assert_eq!(want, tmpl_header);
        assert_eq!("%%template Multiplier_0 [ ff 0  ff 0 ] [ ff 0 ] [3] [ ]", input);
    }

    #[test]
    fn test_parse_ast() {
        let input = ";; Prime value
%%prime 21888242871839275222246405745257275088548364400416034343698204186575808495617

;; Memory of signals
%%signals 4

;; Heap of components
%%components_heap 3

;; Main template
%%start Multiplier_0

;; Component creation mode (implicit/explicit)
%%components explicit

;; Witness (signal list)
%%witness 0 1 2 3

%%template Multiplier_0 [ ff 0  ff 0 ] [ ff 0 ] [3] [ ]
;; store bucket. Line 15
x_0 = get_signal i64.1
;; end of load bucket
x_1 = get_signal i64.2

%%template Multiplier_1 [ ff 0  ff 0 ] [ ff 0 ] [3] [ ]
;; store bucket. Line 15
x_0 = get_signal i64.1
;; end of load bucket
x_1 = get_signal i64.2
";
        let want = AST {
            prime: BigUint::from_str_radix(
                "21888242871839275222246405745257275088548364400416034343698204186575808495617",
                10).unwrap(),
            signals: 4,
            components_heap: 3,
            start: "Multiplier_0".to_string(),
            components_mode: ComponentsMode::Explicit,
            witness: vec![0, 1, 2, 3],
            functions: vec![],
            templates: vec![
                Template {
                    name: "Multiplier_0".to_string(),
                    outputs: vec![Signal::Ff(vec![]), Signal::Ff(vec![])],
                    inputs: vec![Signal::Ff(vec![])],
                    signals_num: 3,
                    components: vec![],
                    body: vec![
                        assign("x_0", &get_signal("1")),
                        assign("x_1", &get_signal("2")),
                    ],
                },
                Template {
                    name: "Multiplier_1".to_string(),
                    outputs: vec![Signal::Ff(vec![]), Signal::Ff(vec![])],
                    inputs: vec![Signal::Ff(vec![])],
                    signals_num: 3,
                    components: vec![],
                    body: vec![
                        assign("x_0", &get_signal("1")),
                        assign("x_1", &get_signal("2")),
                    ],
                },
            ],
        };
        let tmpl_header = consume_parse_result(parse_ast.parse(input));
        assert_eq!(want, tmpl_header);
    }

    #[test]
    fn test_parse_ast_with_function() {
        let input = ";; Prime value
%%prime 21888242871839275222246405745257275088548364400416034343698204186575808495617

;; Memory of signals
%%signals 4

;; Heap of components
%%components_heap 3

;; Main template
%%start Multiplier_0

;; Component creation mode (implicit/explicit)
%%components explicit

;; Witness (signal list)
%%witness 0 1 2 3

%%function f1_0 [] [ i64 0 i64 0 ff 0  ff 2 2 2]
;; compute bucket
;; load bucket
x_2 = ff.mul x_0 x_1
;; end of compute bucket
;; OP(ADD)
x_3 = ff.add x_2 ff.2
;; end of store bucket

%%function f1_1 [i64] [ i64 0 ]
;; compute bucket
;; load bucket
x_2 = ff.mul x_0 x_1
;; end of compute bucket
;; OP(ADD)
x_3 = ff.add x_2 ff.2
;; end of store bucket

%%template Multiplier_0 [ ff 0  ff 0 ] [ ff 0 ] [3] [ ]
;; store bucket. Line 15
x_0 = get_signal i64.1
;; end of load bucket
x_1 = get_signal i64.2

%%template Multiplier_1 [ ff 0  ff 0 ] [ ff 0 ] [3] [ ]
;; store bucket. Line 15
x_0 = get_signal i64.1
;; end of load bucket
x_1 = get_signal i64.2
";
        let want = AST {
            prime: BigUint::from_str_radix(
                "21888242871839275222246405745257275088548364400416034343698204186575808495617",
                10).unwrap(),
            signals: 4,
            components_heap: 3,
            start: "Multiplier_0".to_string(),
            components_mode: ComponentsMode::Explicit,
            witness: vec![0, 1, 2, 3],
            functions: vec![
                Function {
                    name: "f1_0".to_string(),
                    body: vec![
                        assign("x_2", &ff_mul("x_0", "x_1")),
                        assign("x_3", &ff_add("x_2", "2")),
                    ],
                },
                Function {
                    name: "f1_1".to_string(),
                    body: vec![
                        assign("x_2", &ff_mul("x_0", "x_1")),
                        assign("x_3", &ff_add("x_2", "2")),
                    ],
                },
            ],
            templates: vec![
                Template {
                    name: "Multiplier_0".to_string(),
                    outputs: vec![Signal::Ff(vec![]), Signal::Ff(vec![])],
                    inputs: vec![Signal::Ff(vec![])],
                    signals_num: 3,
                    components: vec![],
                    body: vec![
                        assign("x_0", &get_signal("1")),
                        assign("x_1", &get_signal("2")),
                    ],
                },
                Template {
                    name: "Multiplier_1".to_string(),
                    outputs: vec![Signal::Ff(vec![]), Signal::Ff(vec![])],
                    inputs: vec![Signal::Ff(vec![])],
                    signals_num: 3,
                    components: vec![],
                    body: vec![
                        assign("x_0", &get_signal("1")),
                        assign("x_1", &get_signal("2")),
                    ],
                },
            ],
        };
        let tmpl_header = consume_parse_result(parse_ast.parse(input));
        assert_eq!(want, tmpl_header);
    }

    fn consume_parse_result<T, E: std::fmt::Display>(x: winnow::error::Result<T, E>) -> T {
        match x {
            Ok(x) => x,
            Err(e) => {
                println!("Error:\n{}", e);
                panic!();
            }
        }
    }

    #[test]
    fn test_parse_prime() {
        // no newline
        let input = "%%prime 21888242871839275222246405745257275088548364400416034343698204186575808495617";
        let want_err = "\
%%prime 21888242871839275222246405745257275088548364400416034343698204186575808495617
                                                                                     ^
expected newline";
        let r = parse_prime.parse(input);
        assert!(r.is_err());
        let err = r.unwrap_err().to_string();
        assert_eq!(err, want_err);

        // ok
        let mut input = "%%prime 21888242871839275222246405745257275088548364400416034343698204186575808495617
x";
        let want = big_uint("21888242871839275222246405745257275088548364400416034343698204186575808495617");
        let prime = parse_prime.parse_next(&mut input).unwrap();
        assert_eq!(want, prime);
        assert_eq!(input, "x");

        // parse
        let input = "%%prime 21888242871839275222246405745257275088548364400416034343698204186575808495617\n";
        let prime = consume_parse_result(parse_prime.parse(input));
        assert_eq!(want, prime);
    }

    #[test]
    fn test_i64_expression() {
        let input = "i64.load i64.3";
        let want = I64Expr::Load(I64Operand::Literal(3));
        let i64_expr = consume_parse_result(parse_i64_expression.parse(input));
        assert_eq!(want, i64_expr);

        let input = "i64.load x_24";
        let want = I64Expr::Load(I64Operand::Variable("x_24".to_string()));
        let i64_expr = consume_parse_result(parse_i64_expression.parse(input));
        assert_eq!(want, i64_expr);

        let input = "i64.wrap_ff x_8";
        let want = I64Expr::Wrap(Box::new(FfExpr::Variable("x_8".to_string())));
        let i64_expr = consume_parse_result(parse_i64_expression.parse(input));
        assert_eq!(want, i64_expr);

        let input = "i64.mul i64.2 x_9";
        let want = I64Expr::Mul(
            Box::new(I64Expr::Literal(2)),
            Box::new(I64Expr::Variable("x_9".to_string()))
        );
        let i64_expr = consume_parse_result(parse_i64_expression.parse(input));
        assert_eq!(want, i64_expr);

        let input = "i64.le i64.4 x_1";
        let want = I64Expr::Lte(
            Box::new(I64Expr::Literal(4)),
            Box::new(I64Expr::Variable("x_1".to_string()))
        );
        let i64_expr = consume_parse_result(parse_i64_expression.parse(input));
        assert_eq!(want, i64_expr);
    }

    #[test]
    fn test_parse_ff_mreturn() {
        // Test with literal operands
        let input = "ff.mreturn i64.1 i64.32 i64.4";
        let want = Statement::FfMReturn {
            dst: I64Operand::Literal(1),
            src: I64Operand::Literal(32),
            size: I64Operand::Literal(4),
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with variable operands
        let input = "ff.mreturn x_1 x_32 x_34";
        let want = Statement::FfMReturn {
            dst: I64Operand::Variable("x_1".to_string()),
            src: I64Operand::Variable("x_32".to_string()),
            size: I64Operand::Variable("x_34".to_string()),
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with mixed operands
        let input = "ff.mreturn x_1 i64.10 x_size";
        let want = Statement::FfMReturn {
            dst: I64Operand::Variable("x_1".to_string()),
            src: I64Operand::Literal(10),
            size: I64Operand::Variable("x_size".to_string()),
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with comment at end of line
        let mut input = "ff.mreturn x_1 x_49 x_51 ;; returning values
x";
        let want = Statement::FfMReturn {
            dst: I64Operand::Variable("x_1".to_string()),
            src: I64Operand::Variable("x_49".to_string()),
            size: I64Operand::Variable("x_51".to_string()),
        };
        let statement = parse_statement.parse_next(&mut input).unwrap();
        assert_eq!(statement, want);
        assert_eq!("x", input);
    }

    #[test]
    fn test_parse_call_argument() {
        // Test individual call argument parsing first
        let input = "i64.6";
        let want = CallArgument::I64Literal(6);
        let arg = parse_call_argument.parse(input).unwrap();
        assert_eq!(arg, want);

        let input = "ff.3";
        let want = CallArgument::FfLiteral(BigUint::from(3u32));
        let arg = parse_call_argument.parse(input).unwrap();
        assert_eq!(arg, want);

        let input = "ff.memory(i64.2,i64.4)";
        let want = CallArgument::FfMemory {
            addr: I64Operand::Literal(2),
            size: I64Operand::Literal(4),
        };
        let arg = parse_call_argument.parse(input).unwrap();
        assert_eq!(arg, want);

        let input = "i64.memory(i64.10,i64.20)";
        let want = CallArgument::I64Memory {
            addr: I64Operand::Literal(10),
            size: I64Operand::Literal(20),
        };
        let arg = parse_call_argument.parse(input).unwrap();
        assert_eq!(arg, want);
    }

    #[test]
    fn test_parse_variable_assignment() {
        // Test simple variable assignment
        let input = "x_1 = x_2";
        let want = Statement::Assignment {
            name: "x_1".to_string(),
            value: Expr::Variable("x_2".to_string()),
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with spaces
        let input = "my_var   =   other_var";
        let want = Statement::Assignment {
            name: "my_var".to_string(),
            value: Expr::Variable("other_var".to_string()),
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with comment
        let mut input = "result = temp ;; copying temp to result
x";
        let want = Statement::Assignment {
            name: "result".to_string(),
            value: Expr::Variable("temp".to_string()),
        };
        let statement = parse_statement.parse_next(&mut input).unwrap();
        assert_eq!(statement, want);
        assert_eq!("x", input);

        // Test with underscores and numbers
        let input = "x_123 = y_456";
        let want = Statement::Assignment {
            name: "x_123".to_string(),
            value: Expr::Variable("y_456".to_string()),
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);
    }

    #[test]
    fn test_parse_assignment_variable() {
        // Test simple variable assignment
        let input = "x_1 = x_2";
        let want = Statement::Assignment {
            name: "x_1".to_string(),
            value: Expr::Variable("x_2".to_string()),
        };
        let statement = parse_assignment_variable.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with spaces
        let input = "my_var   =   other_var";
        let want = Statement::Assignment {
            name: "my_var".to_string(),
            value: Expr::Variable("other_var".to_string()),
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with comment
        let mut input = "result = temp ;; copying temp to result
x";
        let want = Statement::Assignment {
            name: "result".to_string(),
            value: Expr::Variable("temp".to_string()),
        };
        let statement = parse_statement.parse_next(&mut input).unwrap();
        assert_eq!(statement, want);
        assert_eq!("x", input);

        // Test with underscores and numbers
        let input = "x_123 = y_456";
        let want = Statement::Assignment {
            name: "x_123".to_string(),
            value: Expr::Variable("y_456".to_string()),
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);
    }

    #[test]
    fn test_parse_ff_mcall() {
        // Test with simple case first
        let input = "ff.mcall $f1_0 i64.6";
        let want = Statement::FfMCall {
            name: "f1_0".to_string(),
            args: vec![
                CallArgument::I64Literal(6),
            ],
        };
        let statement = consume_parse_result(parse_statement.parse(input));
        assert_eq!(statement, want);

        // Test with basic example: ff.mcall $f1_0 i64.6 i64.9 ff.3 ff.memory(i64.2,i64.4)
        let input = "ff.mcall $f1_0 i64.6 i64.9 ff.3 ff.memory(i64.2,i64.4)";
        let want = Statement::FfMCall {
            name: "f1_0".to_string(),
            args: vec![
                CallArgument::I64Literal(6),
                CallArgument::I64Literal(9),
                CallArgument::FfLiteral(BigUint::from(3u32)),
                CallArgument::FfMemory {
                    addr: I64Operand::Literal(2),
                    size: I64Operand::Literal(4),
                },
            ],
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with i64.memory
        let input = "ff.mcall $test_func i64.memory(i64.10,i64.20) ff.42";
        let want = Statement::FfMCall {
            name: "test_func".to_string(),
            args: vec![
                CallArgument::I64Memory {
                    addr: I64Operand::Literal(10),
                    size: I64Operand::Literal(20),
                },
                CallArgument::FfLiteral(BigUint::from(42u32)),
            ],
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with variable operands in memory calls
        let input = "ff.mcall $my_function ff.memory(x_addr,x_size) i64.100";
        let want = Statement::FfMCall {
            name: "my_function".to_string(),
            args: vec![
                CallArgument::FfMemory {
                    addr: I64Operand::Variable("x_addr".to_string()),
                    size: I64Operand::Variable("x_size".to_string()),
                },
                CallArgument::I64Literal(100),
            ],
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with mixed memory calls
        let input = "ff.mcall $func i64.memory(i64.5,x_count) ff.memory(x_base,i64.8)";
        let want = Statement::FfMCall {
            name: "func".to_string(),
            args: vec![
                CallArgument::I64Memory {
                    addr: I64Operand::Literal(5),
                    size: I64Operand::Variable("x_count".to_string()),
                },
                CallArgument::FfMemory {
                    addr: I64Operand::Variable("x_base".to_string()),
                    size: I64Operand::Literal(8),
                },
            ],
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with comment at end of line
        let mut input = "ff.mcall $calculator ff.12345 i64.67890 ;; function call with args
x";
        let want = Statement::FfMCall {
            name: "calculator".to_string(),
            args: vec![
                CallArgument::FfLiteral(BigUint::from(12345u32)),
                CallArgument::I64Literal(67890),
            ],
        };
        let statement = parse_statement.parse_next(&mut input).unwrap();
        assert_eq!(statement, want);
        assert_eq!("x", input);

        // Test with no arguments
        let input = "ff.mcall $empty_func";
        let want = Statement::FfMCall {
            name: "empty_func".to_string(),
            args: vec![],
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);

        // Test with large FF literal
        let input = "ff.mcall $big_number_func ff.21888242871839275222246405745257275088548364400416034343698204186575808495617";
        let want = Statement::FfMCall {
            name: "big_number_func".to_string(),
            args: vec![
                CallArgument::FfLiteral(BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap()),
            ],
        };
        let statement = parse_statement.parse(input).unwrap();
        assert_eq!(statement, want);
    }


    fn big_uint(n: &str) -> BigUint {
        BigUint::from_str_radix(n, 10).unwrap()
    }

    fn assign(var_name: &str, expr: &FfExpr) -> TemplateInstruction {
        TemplateInstruction::FfAssignment(FfAssignment {
            dest: var_name.to_string(),
            value: expr.clone(),
        })
    }

    fn assigni(var_name: &str, expr: &I64Expr) -> TemplateInstruction {
        TemplateInstruction::I64Assignment(I64Assignment {
            dest: var_name.to_string(),
            value: expr.clone(),
        })
    }

    fn is_alpha_or_underscore(s: &str) -> bool {
        if let Some(c) = s.chars().next() {
            if c.is_alphabetic() || c == '_' {
                return true;
            }
        }
        false
    }

    fn ff(n: &str) -> FfExpr {
        let is_var = is_alpha_or_underscore(n);

        if is_var {
            FfExpr::Variable(n.to_string())
        } else {
            FfExpr::Literal(big_uint(n))
        }
    }

    fn i64_op(n: &str) -> I64Operand {
        let is_var = is_alpha_or_underscore(n);
        if is_var {
            I64Operand::Variable(n.to_string())
        } else {
            I64Operand::Literal(n.parse().unwrap())
        }
    }

    fn i64_expr(n: &str) -> I64Expr {
        let is_var = is_alpha_or_underscore(n);
        if is_var {
            I64Expr::Variable(n.to_string())
        } else {
            I64Expr::Literal(n.parse().unwrap())
        }
    }

    fn i64_add(op1: &str, op2: &str) -> I64Expr {
        I64Expr::Add(Box::new(i64_expr(op1)), Box::new(i64_expr(op2)))
    }

    fn i64_sub(op1: &str, op2: &str) -> I64Expr {
        I64Expr::Sub(Box::new(i64_expr(op1)), Box::new(i64_expr(op2)))
    }

    fn set_signal(op1: &str, op2: &str) -> TemplateInstruction {
        TemplateInstruction::Statement(Statement::SetSignal {
            idx: i64_op(op1),
            value: ff(op2),
        })
    }

    fn set_cmp_input(cmp_idx: &str, sig_idx: &str, value: &str) -> TemplateInstruction {
        TemplateInstruction::Statement(Statement::SetCmpInput {
            cmp_idx: i64_expr(cmp_idx),
            sig_idx: i64_expr(sig_idx),
            value: ff(value),
        })
    }

    fn ff_div(op1: &str, op2: &str) -> FfExpr {
        FfExpr::FfDiv(
            Box::new(ff(op1)),
            Box::new(ff(op2))
        )
    }

    fn ff_mul(op1: &str, op2: &str) -> FfExpr {
        FfExpr::FfMul(
            Box::new(ff(op1)),
            Box::new(ff(op2))
        )
    }

    fn ff_add(op1: &str, op2: &str) -> FfExpr {
        FfExpr::FfAdd(
            Box::new(ff(op1)),
            Box::new(ff(op2))
        )
    }

    fn get_signal(op1: &str) -> FfExpr {
        FfExpr::GetSignal(i64_op(op1))
    }
}
