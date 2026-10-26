//! Strict DependencyMatcher JSON patterns; node attributes use the token reader.
use super::{ImportError, invalid, json, object, string, token, unsupported};
use crate::{DependencyNode, DependencyOp};

/// Reads one spaCy DependencyMatcher pattern. Node names resolve to earlier
/// indices, and the returned order is the match output order.
pub fn read_dependency_pattern(text: &str) -> Result<Vec<DependencyNode>, ImportError> {
    if text.len() > 16 * 1024 * 1024 {
        return Err(invalid("pattern file too large"));
    }
    let json::Value::Array(values) = json::parse(text)? else {
        return Err(invalid("dependency pattern"));
    };
    if values.is_empty() || values.len() > crate::DependencyMatcher::MAX_NODES {
        return Err(invalid("dependency nodes"));
    }
    let mut names = Vec::new();
    let mut nodes = Vec::new();
    for (i, value) in values.iter().enumerate() {
        let fields = object(value, "dependency node")?;
        for (key, _) in fields {
            if !["LEFT_ID", "REL_OP", "RIGHT_ID", "RIGHT_ATTRS"].contains(&key.as_str()) {
                return Err(unsupported(key));
            }
        }
        let get = |key: &str| fields.iter().find(|(k, _)| k == key).map(|(_, v)| v).ok_or_else(|| invalid(key));
        let name = string(get("RIGHT_ID")?, "RIGHT_ID")?;
        if name.is_empty() || names.contains(&name) {
            return Err(invalid("unique nonempty RIGHT_ID"));
        }
        let attrs = get("RIGHT_ATTRS")?;
        let relation = if i == 0 {
            if fields.len() != 2 {
                return Err(invalid("anchor keys"));
            }
            None
        } else {
            let left = string(get("LEFT_ID")?, "LEFT_ID")?;
            let left = names.iter().position(|n| *n == left).ok_or_else(|| invalid("LEFT_ID must precede RIGHT_ID"))?;
            let operator = string(get("REL_OP")?, "REL_OP")?;
            Some((left, DependencyOp::parse(operator).map_err(|_| unsupported(operator))?))
        };
        nodes.push(DependencyNode { relation, token: token::read(attrs)? });
        names.push(name);
    }
    crate::dependency::validate(&nodes).map_err(|_| invalid("dependency pattern"))?;
    Ok(nodes)
}
