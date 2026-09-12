import Foundation

/// Minimal TOML reader for inferstream config + catalog.
/// Covers strings, ints, bools, arrays, tables, dotted/quoted keys, and
/// array-of-tables. Not a general TOML 1.0 implementation.
public enum TomlValue: Sendable, Equatable {
    case string(String)
    case int(Int64)
    case bool(Bool)
    case array([TomlValue])
    case table(TomlTable)
}

public struct TomlTable: Sendable, Equatable {
    public var values: [String: TomlValue]

    public init(_ values: [String: TomlValue] = [:]) {
        self.values = values
    }

    public subscript(key: String) -> TomlValue? {
        get { values[key] }
        set { values[key] = newValue }
    }

    public func string(_ key: String) -> String? {
        if case .string(let v) = values[key] { return v }
        return nil
    }

    public func int(_ key: String) -> Int64? {
        if case .int(let v) = values[key] { return v }
        return nil
    }

    public func bool(_ key: String) -> Bool? {
        if case .bool(let v) = values[key] { return v }
        return nil
    }

    public func stringArray(_ key: String) -> [String]? {
        guard case .array(let items) = values[key] else { return nil }
        return items.compactMap {
            if case .string(let s) = $0 { return s }
            return nil
        }
    }

    public func table(_ key: String) -> TomlTable? {
        if case .table(let t) = values[key] { return t }
        return nil
    }

    public func tables(_ key: String) -> [TomlTable]? {
        guard case .array(let items) = values[key] else { return nil }
        return items.compactMap {
            if case .table(let t) = $0 { return t }
            return nil
        }
    }
}

public enum TomlError: Error, LocalizedError, Sendable {
    case parse(String)

    public var errorDescription: String? {
        switch self {
        case .parse(let message): message
        }
    }
}

public enum Toml {
    public static func parse(_ text: String) throws -> TomlTable {
        var parser = Parser(text)
        return try parser.parseDocument()
    }

    public static func parseFile(_ path: String) throws -> TomlTable {
        let text = try String(contentsOfFile: path, encoding: .utf8)
        return try parse(text)
    }
}

/// Mutable tree used only while parsing.
private final class Node {
    enum Kind {
        case table([String: Node])
        case array([Node])
        case value(TomlValue)
    }

    var kind: Kind

    init(table: [String: Node] = [:]) { kind = .table(table) }
    init(array: [Node]) { kind = .array(array) }
    init(value: TomlValue) { kind = .value(value) }

    func asTable() -> [String: Node]? {
        if case .table(let t) = kind { return t }
        return nil
    }

    func freeze() -> TomlValue {
        switch kind {
        case .value(let v):
            return v
        case .array(let items):
            return .array(items.map { $0.freeze() })
        case .table(let children):
            var out = TomlTable()
            for (k, child) in children {
                out[k] = child.freeze()
            }
            return .table(out)
        }
    }
}

private struct Parser {
    let chars: [Character]
    var i = 0

    init(_ text: String) {
        self.chars = Array(text)
    }

    mutating func parseDocument() throws -> TomlTable {
        let root = Node()
        var current = root
        while true {
            skipWhitespaceAndComments()
            if atEnd { break }
            if peek() == "[" {
                if peek(1) == "[" {
                    let path = try parseHeader(array: true)
                    current = appendArrayOfTables(root: root, path: path)
                } else {
                    let path = try parseHeader(array: false)
                    current = ensureTable(root: root, path: path)
                }
                continue
            }
            let key = try parseKey()
            skipSpace()
            try expect("=")
            skipSpace()
            let value = try parseValue()
            assign(into: current, path: key, value: Node(value: value))
        }
        guard case .table(let frozen) = root.freeze() else {
            throw TomlError.parse("root is not a table")
        }
        return frozen
    }

    func ensureTable(root: Node, path: [String]) -> Node {
        var cursor = root
        for key in path {
            if case .array(let items) = cursor.kind, let last = items.last {
                cursor = last
            }
            guard case .table(var children) = cursor.kind else {
                let child = Node()
                cursor.kind = .table([key: child])
                cursor = child
                continue
            }
            if let existing = children[key] {
                if case .table = existing.kind {
                    cursor = existing
                    continue
                }
                if case .array(let items) = existing.kind, let last = items.last,
                    case .table = last.kind
                {
                    cursor = last
                    continue
                }
            }
            let child = Node()
            children[key] = child
            cursor.kind = .table(children)
            cursor = child
        }
        return cursor
    }

    func appendArrayOfTables(root: Node, path: [String]) -> Node {
        precondition(!path.isEmpty)
        let parent = path.count == 1 ? root : ensureTable(root: root, path: Array(path.dropLast()))
        let key = path.last!
        let table = Node()
        if case .table(var children) = parent.kind {
            if case .array(var items) = children[key]?.kind {
                items.append(table)
                children[key] = Node(array: items)
            } else {
                children[key] = Node(array: [table])
            }
            parent.kind = .table(children)
        }
        return table
    }

    func assign(into current: Node, path: [String], value: Node) {
        precondition(!path.isEmpty)
        if path.count == 1 {
            if case .table(var children) = current.kind {
                children[path[0]] = value
                current.kind = .table(children)
            }
            return
        }
        let parent = ensureTable(root: current, path: Array(path.dropLast()))
        if case .table(var children) = parent.kind {
            children[path.last!] = value
            parent.kind = .table(children)
        }
    }

    mutating func parseHeader(array: Bool) throws -> [String] {
        try expect("[")
        if array { try expect("[") }
        skipSpace()
        var path: [String] = []
        while true {
            path.append(contentsOf: try parseKey())
            skipSpace()
            if peek() == "." {
                advance()
                skipSpace()
                continue
            }
            break
        }
        try expect("]")
        if array { try expect("]") }
        return path
    }

    mutating func parseKey() throws -> [String] {
        var parts: [String] = []
        while true {
            skipSpace()
            if peek() == "\"" {
                parts.append(try parseBasicString())
            } else if peek() == "'" {
                parts.append(try parseLiteralString())
            } else {
                parts.append(try parseBareKey())
            }
            skipSpace()
            if peek() == "." {
                advance()
                continue
            }
            break
        }
        return parts
    }

    mutating func parseBareKey() throws -> String {
        let start = i
        while let c = peekChar(), c.isLetter || c.isNumber || c == "_" || c == "-" {
            advance()
        }
        if i == start {
            throw TomlError.parse("expected key at \(i)")
        }
        return String(chars[start..<i])
    }

    mutating func parseValue() throws -> TomlValue {
        skipSpace()
        if peek() == "\"" { return .string(try parseBasicString()) }
        if peek() == "'" { return .string(try parseLiteralString()) }
        if peek() == "[" { return .array(try parseArray()) }
        if peek() == "t" || peek() == "f" { return .bool(try parseBool()) }
        if peek() == "-" || peek()?.isNumber == true { return .int(try parseInt()) }
        throw TomlError.parse("unexpected value at \(i)")
    }

    mutating func parseArray() throws -> [TomlValue] {
        try expect("[")
        var items: [TomlValue] = []
        while true {
            skipWhitespaceAndComments()
            if peek() == "]" {
                advance()
                break
            }
            items.append(try parseValue())
            skipWhitespaceAndComments()
            if peek() == "," {
                advance()
                continue
            }
            try expect("]")
            break
        }
        return items
    }

    mutating func parseBool() throws -> Bool {
        if consume("true") { return true }
        if consume("false") { return false }
        throw TomlError.parse("expected bool at \(i)")
    }

    mutating func parseInt() throws -> Int64 {
        let start = i
        if peek() == "-" || peek() == "+" { advance() }
        while peek()?.isNumber == true { advance() }
        let raw = String(chars[start..<i])
        guard let value = Int64(raw) else {
            throw TomlError.parse("bad integer \(raw)")
        }
        return value
    }

    mutating func parseBasicString() throws -> String {
        try expect("\"")
        var out = ""
        while let c = peekChar() {
            if c == "\"" {
                advance()
                return out
            }
            if c == "\\" {
                advance()
                guard let esc = peekChar() else { throw TomlError.parse("unterminated escape") }
                advance()
                switch esc {
                case "n": out.append("\n")
                case "t": out.append("\t")
                case "r": out.append("\r")
                case "\\": out.append("\\")
                case "\"": out.append("\"")
                default: out.append(esc)
                }
                continue
            }
            out.append(c)
            advance()
        }
        throw TomlError.parse("unterminated string")
    }

    mutating func parseLiteralString() throws -> String {
        try expect("'")
        var out = ""
        while let c = peekChar() {
            if c == "'" {
                advance()
                return out
            }
            out.append(c)
            advance()
        }
        throw TomlError.parse("unterminated literal string")
    }

    mutating func skipWhitespaceAndComments() {
        while let c = peekChar() {
            if c == " " || c == "\t" || c == "\n" || c == "\r" {
                advance()
                continue
            }
            if c == "#" {
                while let d = peekChar(), d != "\n" { advance() }
                continue
            }
            break
        }
    }

    mutating func skipSpace() {
        while let c = peekChar(), c == " " || c == "\t" { advance() }
    }

    func peek(_ offset: Int = 0) -> Character? {
        let idx = i + offset
        guard idx < chars.count else { return nil }
        return chars[idx]
    }

    func peekChar() -> Character? { peek() }

    var atEnd: Bool { i >= chars.count }

    mutating func advance() { i += 1 }

    mutating func expect(_ s: Character) throws {
        guard peek() == s else {
            throw TomlError.parse("expected \(s) at \(i), got \(String(describing: peek()))")
        }
        advance()
    }

    mutating func consume(_ word: String) -> Bool {
        let w = Array(word)
        guard i + w.count <= chars.count else { return false }
        if Array(chars[i..<(i + w.count)]) == w {
            i += w.count
            return true
        }
        return false
    }
}
