use num_bigint::BigUint;
use winnow::ascii::{digit1, space0, space1, line_ending, alpha1};
use winnow::error::{ContextError, ErrMode, ParseError, ParserError, StrContext, StrContextValue};
use winnow::{ModalResult};
use winnow::combinator::{alt, preceded, repeat, terminated, seq, dispatch, fail, opt, cut_err, trace, eof, delimited};
use winnow::Parser;
use winnow::token::{literal, take_till, take_while};
use crate::ast::{Assignment, ComponentsMode, FfExpr, I64Operand, Signal, Statement, Template, TemplateInstruction, AST};

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

fn parse_template_body(input: &mut &str) -> ModalResult<Vec<TemplateInstruction>> {
    let mut instructions = Vec::new();
    'outer: while !input.is_empty() {
        let inst = alt((
            parse_assignment.map(|x| Some(TemplateInstruction::Assignment(x))),
            parse_statement.map(|x| Some(TemplateInstruction::Statement(x))),
            parse_empty_line.map(|_| None),
        )).parse_next(input);
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

fn parse_assignment(input: &mut &str) -> ModalResult<Assignment> {
    alt((parse_spr_assignment, parse_var_assignment)).parse_next(input)
}

fn parse_var_assignment(input: &mut &str) -> ModalResult<Assignment> {
    let (var_name, _, _, _, expr, _, _, _) = seq!(
        parse_variable_name,
        space0, literal("="), space0,
        cut_err(parse_expression
            .context(StrContext::Label("expression"))
            .context(StrContext::Expected(StrContextValue::Description("valid expression")))),
        space0,
        opt(parse_eol_comment),
        parse_line_end)
        .parse_next(input)?;
    Ok(Assignment{
        dest: var_name.to_string(),
        value: expr,
    })
}

fn parse_spr_assignment(input: &mut &str) -> ModalResult<Assignment> {
    let (_, _, _, _, ds, _, _, _) = seq!(
        "spr", space0, literal("="), space0,
        cut_err(digit1)
            .context(StrContext::Label("spr assignment"))
            .context(StrContext::Expected(StrContextValue::Description("valid spr assignment"))),
        space0, opt(parse_eol_comment), parse_line_end).parse_next(input)?;
    Ok(Assignment{
        dest: "spr".to_string(),
        value: FfExpr::Literal(ds.parse::<BigUint>().unwrap()),
    })
}

fn parse_statement(input: &mut &str) -> ModalResult<Statement> {
    let s = dispatch! {parse_operator_name;
        "set_signal" => (preceded(space1, parse_i64_operand), preceded(space1, parse_ff_expr))
            .map(|(op1, op2)| Statement::SetSignal{ idx: op1, value: op2 }),
        "set_cmp_input_run" => (
            preceded(space1, parse_i64_operand),
            preceded(space1, parse_i64_operand),
            preceded(space1, parse_ff_expr))
            .map(
                |(op1, op2, op3)| Statement::SetCmpSignalRun{ cmp_idx: op1, sig_idx: op2, value: op3 }),
        "error" => preceded(space1, parse_i64_operand)
            .map(|op1| Statement::Error{ code: op1 }),
        "if" => {
            |i: &mut &str| {
                // Parse the condition
                let (condition, _, _, _) = seq!(
                    preceded(space1, parse_expression),
                    space0,
                    opt(parse_eol_comment),
                    parse_line_end
                ).parse_next(i)?;

                // Parse the if block
                let mut if_block = Vec::new();
                'if_block: while !i.is_empty() {
                    if i.starts_with("else") || i.starts_with("end") {
                        break 'if_block;
                    }

                    let inst = alt((
                        parse_assignment.map(|x| Some(TemplateInstruction::Assignment(x))),
                        parse_statement.map(|x| Some(TemplateInstruction::Statement(x))),
                        parse_empty_line.map(|_| None),
                    )).parse_next(i);

                    match inst {
                        Ok(Some(inst)) => { if_block.push(inst); }
                        Ok(None) => (),
                        Err(err) => {
                            match err {
                                ErrMode::Cut(_) => { return Err(err); }
                                _ => { break 'if_block; }
                            }
                        }
                    }
                }

                // Parse the optional else block
                let mut else_block = Vec::new();
                if i.starts_with("else") {
                    // Consume the "else" keyword
                    let _ = (literal("else"), space0, opt(parse_eol_comment), parse_line_end).parse_next(i)?;

                    'else_block: while !i.is_empty() {
                        if i.starts_with("end") {
                            break 'else_block;
                        }

                        let inst = alt((
                            parse_assignment.map(|x| Some(TemplateInstruction::Assignment(x))),
                            parse_statement.map(|x| Some(TemplateInstruction::Statement(x))),
                            parse_empty_line.map(|_| None),
                        )).parse_next(i);

                        match inst {
                            Ok(Some(inst)) => { else_block.push(inst); }
                            Ok(None) => (),
                            Err(err) => {
                                match err {
                                    ErrMode::Cut(_) => { return Err(err); }
                                    _ => { break 'else_block; }
                                }
                            }
                        }
                    }
                }

                // Consume the "end" keyword
                let _ = (literal("end"), space0, opt(parse_eol_comment), parse_line_end).parse_next(i)?;

                Ok(Statement::Branch {
                    condition,
                    if_block,
                    else_block,
                })
            }
        },
        _ => fail::<_, Statement, _>,
    }
        .parse_next(input)?;

    // For set_signal, we need to parse the line end
    if let Statement::SetSignal { .. } = s {
        (space0, opt(parse_eol_comment), parse_line_end).parse_next(input)?;
    }

    Ok(s)
}

fn parse_expression(input: &mut &str) -> ModalResult<FfExpr> {
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
            _ => fail::<_, FfExpr, _>,
        },
        // Try to parse as a literal
        parse_ff_literal.map(FfExpr::Literal),
        // Try to parse as a variable
        parse_variable_name.map(|name| FfExpr::Variable(name.to_string())),
    )).parse_next(input)?;

    Ok(result)
}

fn parse_eol_comment(i: &mut &str) -> ModalResult<()>
{
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
    use winnow::error::{ContextError};
    use crate::parser::{parse_var_assignment, parse_ast, parse_bus_signal, parse_eol_comment, parse_expression, parse_ff_signal, parse_i64_literal, parse_i64_operand, parse_operator_name, parse_prime, parse_statement, parse_template, parse_template_body, parse_variable_name};
    use winnow::{Parser};
    use winnow::token::{take_till};
    use crate::ast::{Assignment, ComponentsMode, FfExpr, I64Operand, Signal, Statement, Template, TemplateInstruction, AST};

    #[test]
    fn test_parse_branch() {
        let mut input = r#"if x_1
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

        let statement = parse_statement.parse(&mut input).unwrap();

        let want = Statement::Branch {
            condition: ff("x_1"),
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
        let mut input = r#"if get_signal x_1
x_2 = get_signal i64.1
x_3 = ff.div ff.1 x_2
set_signal i64.2 x_3
else
set_signal i64.2 ff.0
end
"#;

        let statement = parse_statement.parse(&mut input).unwrap();

        let want = Statement::Branch {
            condition: get_signal("x_1"),
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

        let mut input = "i64.93841982938198593829123 = get_signal";
        let want_err = "\
i64.93841982938198593829123 = get_signal
    ^
invalid i64 literal
expected valid i64 value";
        let op = parse_i64_operand.parse(&mut input)
            .unwrap_err().to_string();

        assert_eq!(op, want_err);
    }

    #[test]
    fn test_parse_expression() {
        let mut input = "get_signal x_50";
        let want = get_signal("x_50");
        let op = parse_expression.parse(&mut input).unwrap();
        assert_eq!(op, want);

        let mut input = "get_signal i64.2";
        let want = get_signal("2");
        let op = parse_expression.parse(&mut input).unwrap();
        assert_eq!(op, want);

        let mut input = "ff.div ff.2 v_3";
        let want = ff_div("2", "v_3");
        let op = parse_expression.parse(&mut input).unwrap();
        assert_eq!(op, want);

        let mut input = "v_3";
        let want = ff("v_3");
        let op = parse_expression.parse(&mut input).unwrap();
        assert_eq!(op, want);

        let mut input = "ff.5";
        let want = ff("5");
        let op = parse_expression.parse(&mut input).unwrap();
        assert_eq!(op, want);
    }

    #[test]
    fn test_assignment() {
        let mut input = "x_4 = get_signal i64.2";
        let want = Assignment{
            dest: "x_4".to_string(),
            value: get_signal("2"),
        };
        let a = parse_var_assignment.parse(&mut input).unwrap();
        assert_eq!(a, want);

        let mut input = "x_4 = get_signal i64.2 2";
        assert!(parse_var_assignment.parse_next(&mut input).is_err());

        let mut input = "x_4 = get_signal i64.2\nxxx";
        let want = Assignment{
            dest: "x_4".to_string(),
            value: get_signal("2"),
        };
        let a = parse_var_assignment.parse_next(&mut input).unwrap();
        assert_eq!(a, want);
        assert_eq!(input, "xxx");

        let mut input = "x_4 = get_signal i64.2;;\nxxx";
        let want = Assignment{
            dest: "x_4".to_string(),
            value: get_signal("2"),
        };
        let a = parse_var_assignment.parse_next(&mut input).unwrap();
        assert_eq!(a, want);
        assert_eq!(input, "xxx");

        let mut input = "x_4 = get_signal i64.2 ;; comment \nxxx";
        let want = Assignment{
            dest: "x_4".to_string(),
            value: get_signal("2"),
        };
        let a = parse_var_assignment.parse_next(&mut input).unwrap();
        assert_eq!(a, want);
        assert_eq!(input, "xxx");

        // test consumes a new line
        let mut input = "x_1 = get_signal i64.2
;; OP(MUL)
x_2 = ff.mul x_0 x_1";
        let want = Assignment{
            dest: "x_1".to_string(),
            value: get_signal("2"),
        };
        let a = parse_var_assignment.parse_next(&mut input).unwrap();
        assert_eq!(a, want);
        let want_left = ";; OP(MUL)
x_2 = ff.mul x_0 x_1";
        assert_eq!(input, want_left);
    }

    #[test]
    fn test_statement() {
        let mut input = "set_signal i64.0 x_3";
        let want = Statement::SetSignal {
            idx: I64Operand::Literal(0),
            value: ff("x_3"),
        };
        let a = parse_statement.parse(&mut input).unwrap();
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
    }

    #[test]
    fn test_parse_i64_literal_error() {
        let mut input = "i64.93841982938198593829123 = get_signal";
        let op = parse_i64_literal.parse(&mut input).unwrap_err();
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
    fn test_4() {
        let mut input = ";; xxx";
        let x = take_till::<_, _, ContextError>(1.., ['x', '\r'])
            .parse_next(&mut input).unwrap();
        println!("{}", x);
        println!("{}", input);
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
    fn test_parse_template() {
        let mut input = "%%template Multiplier_0 [ ff 0  ff 0 ] [ ff 0 ] [3] [ ]
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
        let tmpl_header = parse_template.parse_next(&mut input).unwrap();
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
        let mut input = ";; Prime value
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
        let tmpl_header = consume_parse_result(parse_ast.parse(&mut input));
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
        let mut input = "%%prime 21888242871839275222246405745257275088548364400416034343698204186575808495617";
        let want_err = "\
%%prime 21888242871839275222246405745257275088548364400416034343698204186575808495617
                                                                                     ^
expected newline";
        let r = parse_prime.parse(&mut input);
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
        let mut input = "%%prime 21888242871839275222246405745257275088548364400416034343698204186575808495617\n";
        let prime = consume_parse_result(parse_prime.parse(&mut input));
        assert_eq!(want, prime);
    }

    fn big_uint(n: &str) -> BigUint {
        BigUint::from_str_radix(n, 10).unwrap()
    }

    fn assign(var_name: &str, expr: &FfExpr) -> TemplateInstruction {
        TemplateInstruction::Assignment(Assignment{
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

    fn set_signal(op1: &str, op2: &str) -> TemplateInstruction {
        TemplateInstruction::Statement(Statement::SetSignal {
            idx: i64_op(op1),
            value: ff(op2),
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
