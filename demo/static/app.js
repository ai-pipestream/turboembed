// Built by scripts/build-web.sh from demo/web/src; do not edit.

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/reflect/names.js
function protoCamelCase(snakeCase) {
  let capNext = false;
  const b = [];
  for (let i = 0; i < snakeCase.length; i++) {
    let c = snakeCase.charAt(i);
    switch (c) {
      case "_":
        capNext = true;
        break;
      case "0":
      case "1":
      case "2":
      case "3":
      case "4":
      case "5":
      case "6":
      case "7":
      case "8":
      case "9":
        b.push(c);
        capNext = false;
        break;
      default:
        if (capNext) {
          capNext = false;
          c = c.toUpperCase();
        }
        b.push(c);
        break;
    }
  }
  return b.join("");
}
function protoSnakeCase(lowerCamelCase) {
  return lowerCamelCase.replace(/[A-Z]/g, (letter) => "_" + letter.toLowerCase());
}
var reservedObjectProperties = /* @__PURE__ */ new Set([
  // names reserved by JavaScript
  "constructor",
  "toString",
  "toJSON",
  "valueOf"
]);
function safeObjectProperty(name) {
  return reservedObjectProperties.has(name) ? name + "$" : name;
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/wire/varint.js
function varint64read() {
  const buf = this.buf;
  let pos = this.pos;
  let lo = 0;
  let hi = 0;
  for (let shift = 0; shift < 28; shift += 7) {
    const b = buf[pos++];
    lo |= (b & 127) << shift;
    if ((b & 128) == 0) {
      this.pos = pos;
      this.assertBounds();
      this.varint64Lo = lo;
      this.varint64Hi = hi;
      return;
    }
  }
  const middleByte = buf[pos++];
  lo |= (middleByte & 15) << 28;
  hi = (middleByte & 112) >> 4;
  if ((middleByte & 128) == 0) {
    this.pos = pos;
    this.assertBounds();
    this.varint64Lo = lo;
    this.varint64Hi = hi;
    return;
  }
  for (let shift = 3; shift <= 31; shift += 7) {
    const b = buf[pos++];
    hi |= (b & 127) << shift;
    if ((b & 128) == 0) {
      this.pos = pos;
      this.assertBounds();
      this.varint64Lo = lo;
      this.varint64Hi = hi;
      return;
    }
  }
  throw new Error("invalid varint");
}
var TWO_PWR_32_DBL = 4294967296;
function int64FromString(dec) {
  const minus = dec[0] === "-";
  if (minus) {
    dec = dec.slice(1);
  }
  const base2 = 1e6;
  let lowBits = 0;
  let highBits = 0;
  function add1e6digit(begin, end) {
    const digit1e6 = Number(dec.slice(begin, end));
    highBits *= base2;
    lowBits = lowBits * base2 + digit1e6;
    if (lowBits >= TWO_PWR_32_DBL) {
      highBits = highBits + (lowBits / TWO_PWR_32_DBL | 0);
      lowBits = lowBits % TWO_PWR_32_DBL;
    }
  }
  add1e6digit(-24, -18);
  add1e6digit(-18, -12);
  add1e6digit(-12, -6);
  add1e6digit(-6);
  return minus ? negate(lowBits, highBits) : newBits(lowBits, highBits);
}
function int64ToString(lo, hi) {
  let bits = newBits(lo, hi);
  const negative = bits.hi & 2147483648;
  if (negative) {
    bits = negate(bits.lo, bits.hi);
  }
  const result = uInt64ToString(bits.lo, bits.hi);
  return negative ? "-" + result : result;
}
function uInt64ToString(lo, hi) {
  ({ lo, hi } = toUnsigned(lo, hi));
  if (hi <= 2097151) {
    return String(TWO_PWR_32_DBL * hi + lo);
  }
  const low = lo & 16777215;
  const mid = (lo >>> 24 | hi << 8) & 16777215;
  const high = hi >> 16 & 65535;
  let digitA = low + mid * 6777216 + high * 6710656;
  let digitB = mid + high * 8147497;
  let digitC = high * 2;
  const base2 = 1e7;
  if (digitA >= base2) {
    digitB += Math.floor(digitA / base2);
    digitA %= base2;
  }
  if (digitB >= base2) {
    digitC += Math.floor(digitB / base2);
    digitB %= base2;
  }
  return digitC.toString() + decimalFrom1e7WithLeadingZeros(digitB) + decimalFrom1e7WithLeadingZeros(digitA);
}
function toUnsigned(lo, hi) {
  return { lo: lo >>> 0, hi: hi >>> 0 };
}
function newBits(lo, hi) {
  return { lo: lo | 0, hi: hi | 0 };
}
function negate(lowBits, highBits) {
  highBits = ~highBits;
  if (lowBits) {
    lowBits = ~lowBits + 1;
  } else {
    highBits += 1;
  }
  return newBits(lowBits, highBits);
}
var decimalFrom1e7WithLeadingZeros = (digit1e7) => {
  const partial = String(digit1e7);
  return "0000000".slice(partial.length) + partial;
};
function varint32write(value, bytes2) {
  if (value >>> 0 < 128) {
    bytes2.push(value);
    return;
  }
  if (value >= 0) {
    while (value > 127) {
      bytes2.push(value & 127 | 128);
      value = value >>> 7;
    }
    bytes2.push(value);
  } else {
    for (let i = 0; i < 9; i++) {
      bytes2.push(value & 127 | 128);
      value = value >> 7;
    }
    bytes2.push(1);
  }
}
function varint32read() {
  let b = this.buf[this.pos++];
  if ((b & 128) === 0) {
    this.assertBounds();
    return b;
  }
  let result = b & 127;
  b = this.buf[this.pos++];
  result |= (b & 127) << 7;
  if ((b & 128) === 0) {
    this.assertBounds();
    return result;
  }
  b = this.buf[this.pos++];
  result |= (b & 127) << 14;
  if ((b & 128) === 0) {
    this.assertBounds();
    return result;
  }
  b = this.buf[this.pos++];
  result |= (b & 127) << 21;
  if ((b & 128) === 0) {
    this.assertBounds();
    return result;
  }
  b = this.buf[this.pos++];
  result |= (b & 15) << 28;
  for (let readBytes = 5; (b & 128) !== 0 && readBytes < 10; readBytes++)
    b = this.buf[this.pos++];
  if ((b & 128) !== 0)
    throw new Error("invalid varint");
  this.assertBounds();
  return result >>> 0;
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/proto-int64.js
var protoInt64 = /* @__PURE__ */ makeInt64Support();
function makeInt64Support() {
  const dv = new DataView(new ArrayBuffer(8));
  const ok = typeof BigInt === "function" && typeof dv.getBigInt64 === "function" && typeof dv.getBigUint64 === "function" && typeof dv.setBigInt64 === "function" && typeof dv.setBigUint64 === "function" && (!!globalThis.Deno || !!globalThis.Bun || typeof process != "object" || typeof process.env != "object" || process.env.BUF_BIGINT_DISABLE !== "1");
  if (ok) {
    const MIN = BigInt("-9223372036854775808");
    const MAX = BigInt("9223372036854775807");
    const UMIN = BigInt("0");
    const UMAX = BigInt("18446744073709551615");
    return {
      zero: BigInt(0),
      supported: true,
      parse(value) {
        const bi = typeof value == "bigint" ? value : BigInt(value);
        if (bi > MAX || bi < MIN) {
          throw new Error(`invalid int64: ${value}`);
        }
        return bi;
      },
      uParse(value) {
        const bi = typeof value == "bigint" ? value : BigInt(value);
        if (bi > UMAX || bi < UMIN) {
          throw new Error(`invalid uint64: ${value}`);
        }
        return bi;
      },
      enc(value) {
        dv.setBigInt64(0, this.parse(value), true);
        return {
          lo: dv.getInt32(0, true),
          hi: dv.getInt32(4, true)
        };
      },
      uEnc(value) {
        dv.setBigInt64(0, this.uParse(value), true);
        return {
          lo: dv.getInt32(0, true),
          hi: dv.getInt32(4, true)
        };
      },
      dec(lo, hi) {
        dv.setInt32(0, lo, true);
        dv.setInt32(4, hi, true);
        return dv.getBigInt64(0, true);
      },
      uDec(lo, hi) {
        dv.setInt32(0, lo, true);
        dv.setInt32(4, hi, true);
        return dv.getBigUint64(0, true);
      }
    };
  }
  return {
    zero: "0",
    supported: false,
    parse(value) {
      if (typeof value != "string") {
        value = value.toString();
      }
      assertInt64String(value);
      return value;
    },
    uParse(value) {
      if (typeof value != "string") {
        value = value.toString();
      }
      assertUInt64String(value);
      return value;
    },
    enc(value) {
      if (typeof value != "string") {
        value = value.toString();
      }
      assertInt64String(value);
      return int64FromString(value);
    },
    uEnc(value) {
      if (typeof value != "string") {
        value = value.toString();
      }
      assertUInt64String(value);
      return int64FromString(value);
    },
    dec(lo, hi) {
      return int64ToString(lo, hi);
    },
    uDec(lo, hi) {
      return uInt64ToString(lo, hi);
    }
  };
}
function assertInt64String(value) {
  if (!/^-?[0-9]+$/.test(value)) {
    throw new Error("invalid int64: " + value);
  }
}
function assertUInt64String(value) {
  if (!/^[0-9]+$/.test(value)) {
    throw new Error("invalid uint64: " + value);
  }
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/descriptors.js
var ScalarType;
(function(ScalarType2) {
  ScalarType2[ScalarType2["DOUBLE"] = 1] = "DOUBLE";
  ScalarType2[ScalarType2["FLOAT"] = 2] = "FLOAT";
  ScalarType2[ScalarType2["INT64"] = 3] = "INT64";
  ScalarType2[ScalarType2["UINT64"] = 4] = "UINT64";
  ScalarType2[ScalarType2["INT32"] = 5] = "INT32";
  ScalarType2[ScalarType2["FIXED64"] = 6] = "FIXED64";
  ScalarType2[ScalarType2["FIXED32"] = 7] = "FIXED32";
  ScalarType2[ScalarType2["BOOL"] = 8] = "BOOL";
  ScalarType2[ScalarType2["STRING"] = 9] = "STRING";
  ScalarType2[ScalarType2["BYTES"] = 12] = "BYTES";
  ScalarType2[ScalarType2["UINT32"] = 13] = "UINT32";
  ScalarType2[ScalarType2["SFIXED32"] = 15] = "SFIXED32";
  ScalarType2[ScalarType2["SFIXED64"] = 16] = "SFIXED64";
  ScalarType2[ScalarType2["SINT32"] = 17] = "SINT32";
  ScalarType2[ScalarType2["SINT64"] = 18] = "SINT64";
})(ScalarType || (ScalarType = {}));

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/reflect/scalar.js
function scalarZeroValue(type, longAsString) {
  switch (type) {
    case ScalarType.STRING:
      return "";
    case ScalarType.BOOL:
      return false;
    case ScalarType.DOUBLE:
    case ScalarType.FLOAT:
      return 0;
    case ScalarType.INT64:
    case ScalarType.UINT64:
    case ScalarType.SFIXED64:
    case ScalarType.FIXED64:
    case ScalarType.SINT64:
      return longAsString ? "0" : protoInt64.zero;
    case ScalarType.BYTES:
      return new Uint8Array(0);
    default:
      return 0;
  }
}
function isScalarZeroValue(type, value) {
  switch (type) {
    case ScalarType.BOOL:
      return value === false;
    case ScalarType.STRING:
      return value === "";
    case ScalarType.BYTES:
      return value instanceof Uint8Array && !value.byteLength;
    case ScalarType.DOUBLE:
    case ScalarType.FLOAT:
      return Object.is(value, 0);
    default:
      return value == 0;
  }
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/reflect/unsafe.js
var IMPLICIT = 2;
var unsafeLocal = /* @__PURE__ */ Symbol.for("reflect unsafe local");
function unsafeOneofCase(target, oneof) {
  const c = target[oneof.localName].case;
  if (c === void 0) {
    return c;
  }
  return oneof.fields.find((f) => f.localName === c);
}
function unsafeIsSet(target, field) {
  const name = field.localName;
  if (field.oneof) {
    return target[field.oneof.localName].case === name;
  }
  if (field.presence != IMPLICIT) {
    return target[name] !== void 0 && Object.prototype.hasOwnProperty.call(target, name);
  }
  switch (field.fieldKind) {
    case "list":
      return target[name].length > 0;
    case "map":
      return Object.keys(target[name]).length > 0;
    case "scalar":
      return !isScalarZeroValue(field.scalar, target[name]);
    case "enum":
      return target[name] !== field.enum.values[0].number;
  }
  throw new Error("message field with implicit presence");
}
function unsafeIsSetExplicit(target, localName) {
  return Object.prototype.hasOwnProperty.call(target, localName) && target[localName] !== void 0;
}
function unsafeGet(target, field) {
  if (field.oneof) {
    const oneof = target[field.oneof.localName];
    if (oneof.case === field.localName) {
      return oneof.value;
    }
    return void 0;
  }
  return target[field.localName];
}
function unsafeSet(target, field, value) {
  if (field.oneof) {
    target[field.oneof.localName] = {
      case: field.localName,
      value
    };
  } else {
    target[field.localName] = value;
  }
}
function setOwn(obj, key, value) {
  if (key === "__proto__") {
    Object.defineProperty(obj, "__proto__", {
      value,
      writable: true,
      enumerable: true,
      configurable: true
    });
  } else {
    obj[key] = value;
  }
}
function getOwn(obj, key) {
  return Object.prototype.hasOwnProperty.call(obj, key) ? obj[key] : void 0;
}
function unsafeClear(target, field) {
  const name = field.localName;
  if (field.oneof) {
    const oneofLocalName = field.oneof.localName;
    if (target[oneofLocalName].case === name) {
      target[oneofLocalName] = { case: void 0 };
    }
  } else if (field.presence != IMPLICIT) {
    delete target[name];
  } else {
    switch (field.fieldKind) {
      case "map":
        target[name] = {};
        break;
      case "list":
        target[name] = [];
        break;
      case "enum":
        target[name] = field.enum.values[0].number;
        break;
      case "scalar":
        target[name] = scalarZeroValue(field.scalar, field.longAsString);
        break;
    }
  }
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/codegenv2/restore-json-names.js
function restoreJsonNames(message) {
  for (const f of message.field) {
    if (!unsafeIsSetExplicit(f, "jsonName")) {
      f.jsonName = protoCamelCase(f.name);
    }
  }
  message.nestedType.forEach(restoreJsonNames);
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/wire/text-format.js
function parseTextFormatEnumValue(descEnum, value) {
  const enumValue = descEnum.values.find((v) => v.name === value);
  if (!enumValue) {
    throw new Error(`cannot parse ${descEnum} default value: ${value}`);
  }
  return enumValue.number;
}
function parseTextFormatScalarValue(type, value) {
  switch (type) {
    case ScalarType.STRING:
      return value;
    case ScalarType.BYTES: {
      const u = unescapeBytesDefaultValue(value);
      if (u === false) {
        throw new Error(`cannot parse ${ScalarType[type]} default value: ${value}`);
      }
      return u;
    }
    case ScalarType.INT64:
    case ScalarType.SFIXED64:
    case ScalarType.SINT64:
      return protoInt64.parse(value);
    case ScalarType.UINT64:
    case ScalarType.FIXED64:
      return protoInt64.uParse(value);
    case ScalarType.DOUBLE:
    case ScalarType.FLOAT:
      switch (value) {
        case "inf":
          return Number.POSITIVE_INFINITY;
        case "-inf":
          return Number.NEGATIVE_INFINITY;
        case "nan":
          return Number.NaN;
        default:
          return parseFloat(value);
      }
    case ScalarType.BOOL:
      return value === "true";
    case ScalarType.INT32:
    case ScalarType.UINT32:
    case ScalarType.SINT32:
    case ScalarType.FIXED32:
    case ScalarType.SFIXED32:
      return parseInt(value, 10);
  }
}
function unescapeBytesDefaultValue(str) {
  const b = [];
  const input2 = {
    tail: str,
    c: "",
    next() {
      if (this.tail.length == 0) {
        return false;
      }
      this.c = this.tail[0];
      this.tail = this.tail.substring(1);
      return true;
    },
    take(n) {
      if (this.tail.length >= n) {
        const r = this.tail.substring(0, n);
        this.tail = this.tail.substring(n);
        return r;
      }
      return false;
    }
  };
  while (input2.next()) {
    switch (input2.c) {
      case "\\":
        if (input2.next()) {
          switch (input2.c) {
            case "\\":
              b.push(input2.c.charCodeAt(0));
              break;
            case "b":
              b.push(8);
              break;
            case "f":
              b.push(12);
              break;
            case "n":
              b.push(10);
              break;
            case "r":
              b.push(13);
              break;
            case "t":
              b.push(9);
              break;
            case "v":
              b.push(11);
              break;
            case "0":
            case "1":
            case "2":
            case "3":
            case "4":
            case "5":
            case "6":
            case "7": {
              const s = input2.c;
              const t = input2.take(2);
              if (t === false) {
                return false;
              }
              const n = parseInt(s + t, 8);
              if (Number.isNaN(n)) {
                return false;
              }
              b.push(n);
              break;
            }
            case "x": {
              const s = input2.c;
              const t = input2.take(2);
              if (t === false) {
                return false;
              }
              const n = parseInt(s + t, 16);
              if (Number.isNaN(n)) {
                return false;
              }
              b.push(n);
              break;
            }
            case "u": {
              const s = input2.c;
              const t = input2.take(4);
              if (t === false) {
                return false;
              }
              const n = parseInt(s + t, 16);
              if (Number.isNaN(n)) {
                return false;
              }
              const chunk = new Uint8Array(4);
              const view = new DataView(chunk.buffer);
              view.setInt32(0, n, true);
              b.push(chunk[0], chunk[1], chunk[2], chunk[3]);
              break;
            }
            case "U": {
              const s = input2.c;
              const t = input2.take(8);
              if (t === false) {
                return false;
              }
              const tc = protoInt64.uEnc(s + t);
              const chunk = new Uint8Array(8);
              const view = new DataView(chunk.buffer);
              view.setInt32(0, tc.lo, true);
              view.setInt32(4, tc.hi, true);
              b.push(chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]);
              break;
            }
          }
        }
        break;
      default:
        b.push(input2.c.charCodeAt(0));
    }
  }
  return new Uint8Array(b);
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/reflect/nested-types.js
function* nestedTypes(desc) {
  switch (desc.kind) {
    case "file":
      for (const message of desc.messages) {
        yield message;
        yield* nestedTypes(message);
      }
      yield* desc.enums;
      yield* desc.services;
      yield* desc.extensions;
      break;
    case "message":
      for (const message of desc.nestedMessages) {
        yield message;
        yield* nestedTypes(message);
      }
      yield* desc.nestedEnums;
      yield* desc.nestedExtensions;
      break;
  }
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/wire/text-encoding.js
var te;
function configureTextEncoding(textEncoding) {
  var _a;
  te = Object.assign(Object.assign({}, textEncoding), { encodeUtf8Into: (_a = textEncoding.encodeUtf8Into) !== null && _a !== void 0 ? _a : emulateEncodeInto(textEncoding.encodeUtf8.bind(textEncoding)) });
}
function getTextEncoding() {
  if (!te) {
    const globals = globalThis;
    if (!globals.TextEncoder || !globals.TextDecoder) {
      throw new Error("encoding API missing: install TextEncoder and TextDecoder on globalThis");
    }
    const textEncoder = new globals.TextEncoder();
    const textDecoder = new globals.TextDecoder();
    let textDecoderStrict;
    const config = {
      encodeUtf8(text) {
        return textEncoder.encode(text);
      },
      decodeUtf8(bytes2, strict) {
        if (strict) {
          if (!textDecoderStrict) {
            textDecoderStrict = new globals.TextDecoder("utf-8", {
              fatal: true
            });
          }
          return textDecoderStrict.decode(bytes2);
        }
        return textDecoder.decode(bytes2);
      },
      checkUtf8(text) {
        try {
          encodeURIComponent(text);
          return true;
        } catch (_) {
          return false;
        }
      }
    };
    if (textEncoder.encodeInto) {
      config.encodeUtf8Into = textEncoder.encodeInto.bind(textEncoder);
    }
    const nativeStringIsWellFormed = String.prototype.isWellFormed;
    if (nativeStringIsWellFormed) {
      config.checkUtf8 = (text) => {
        return nativeStringIsWellFormed.call(text);
      };
    }
    configureTextEncoding(config);
  }
  return te;
}
function emulateEncodeInto(encodeUtf8) {
  return (text, dest) => {
    const bytes2 = encodeUtf8(text);
    dest.set(bytes2);
    return { written: bytes2.byteLength };
  };
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/wire/binary-encoding.js
var WireType;
(function(WireType2) {
  WireType2[WireType2["Varint"] = 0] = "Varint";
  WireType2[WireType2["Bit64"] = 1] = "Bit64";
  WireType2[WireType2["LengthDelimited"] = 2] = "LengthDelimited";
  WireType2[WireType2["StartGroup"] = 3] = "StartGroup";
  WireType2[WireType2["EndGroup"] = 4] = "EndGroup";
  WireType2[WireType2["Bit32"] = 5] = "Bit32";
})(WireType || (WireType = {}));
var UINT32_MAX = 4294967295;
var INT32_MAX = 2147483647;
var INT32_MIN = -2147483648;
var BinaryWriter = class {
  constructor(encodeUtf8) {
    this.stackPos = [];
    this.encodeUtf8Into = encodeUtf8 ? emulateEncodeInto(encodeUtf8) : getTextEncoding().encodeUtf8Into;
    this.buffer = EMPTY_BUFFER;
    this.viewCache = EMPTY_VIEW;
    this.pos = 0;
  }
  ensureCapacity(size2) {
    const required = this.pos + size2;
    if (required > this.buffer.length) {
      let newLen = this.buffer.length || INITIAL_SIZE;
      while (newLen < required)
        newLen *= 2;
      const newBuf = new Uint8Array(newLen);
      if (this.pos > 0)
        newBuf.set(this.buffer);
      this.buffer = newBuf;
    }
  }
  /**
   * The DataView over `buffer`, rebuilt only if the buffer has grown since it
   * was last used.
   */
  view() {
    const bytes2 = this.buffer;
    const view = this.viewCache;
    if (view.byteLength === bytes2.byteLength)
      return view;
    const newView = new DataView(bytes2.buffer);
    this.viewCache = newView;
    return newView;
  }
  /**
   * Return all bytes written and reset this writer.
   */
  finish() {
    const result = this.buffer.slice(0, this.pos);
    this.pos = 0;
    this.stackPos = [];
    return result;
  }
  /**
   * Start a new fork for length-delimited data like a message
   * or a packed repeated field.
   *
   * Must be joined later with `join()`.
   */
  fork() {
    this.stackPos.push(this.pos);
    this.ensureCapacity(DEFAULT_LEN_PREFIX_SIZE);
    this.buffer[this.pos++] = 0;
    return this;
  }
  /**
   * Join the last fork. Write its length and bytes, then
   * return to the previous state.
   */
  join() {
    const forkPos = this.stackPos.pop();
    if (forkPos === void 0)
      throw new Error("invalid state, fork stack empty");
    const len = this.pos - forkPos - DEFAULT_LEN_PREFIX_SIZE;
    const lenPrefixSize = varint32Size(len);
    if (lenPrefixSize > DEFAULT_LEN_PREFIX_SIZE) {
      this.ensureCapacity(lenPrefixSize - DEFAULT_LEN_PREFIX_SIZE);
      this.buffer.copyWithin(forkPos + lenPrefixSize, forkPos + DEFAULT_LEN_PREFIX_SIZE, this.pos);
    }
    this.pos = forkPos;
    this.uint32(len);
    this.pos += len;
    return this;
  }
  /**
   * Writes a tag (field number and wire type).
   *
   * Equivalent to `uint32( (fieldNo << 3 | type) >>> 0 )`.
   *
   * Generated code should compute the tag ahead of time and call `uint32()`.
   */
  tag(fieldNo, type) {
    return this.uint32((fieldNo << 3 | type) >>> 0);
  }
  /**
   * Write a chunk of raw bytes.
   */
  raw(chunk) {
    this.ensureCapacity(chunk.length);
    this.buffer.set(chunk, this.pos);
    this.pos += chunk.length;
    return this;
  }
  /**
   * Write a `uint32` value, an unsigned 32 bit varint.
   */
  uint32(value) {
    assertUInt32(value);
    this.ensureCapacity(5);
    if (value < 128) {
      this.buffer[this.pos++] = value;
      return this;
    }
    while (value > 127) {
      this.buffer[this.pos++] = value & 127 | 128;
      value >>>= 7;
    }
    this.buffer[this.pos++] = value;
    return this;
  }
  /**
   * Write a `int32` value, a signed 32 bit varint.
   */
  int32(value) {
    assertInt32(value);
    if (value >= 0) {
      return this.uint32(value);
    }
    this.ensureCapacity(10);
    for (let i = 0; i < 9; i++) {
      this.buffer[this.pos++] = value & 127 | 128;
      value >>= 7;
    }
    this.buffer[this.pos++] = 1;
    return this;
  }
  /**
   * Write a `bool` value, a varint.
   */
  bool(value) {
    this.ensureCapacity(1);
    this.buffer[this.pos++] = value ? 1 : 0;
    return this;
  }
  /**
   * Write a `bytes` value, length-delimited arbitrary data.
   */
  bytes(value) {
    this.uint32(value.byteLength);
    return this.raw(value);
  }
  /**
   * Write a `string` value, length-delimited data converted to UTF-8 text.
   */
  string(value) {
    if (typeof value !== "string") {
      value = String(value);
    }
    const len = value.length;
    if (len <= ASCII_MAX_LENGTH) {
      this.ensureCapacity(len + 1);
      const ascii = this.buffer;
      let pos = this.pos;
      ascii[pos++] = len;
      let i = 0;
      for (; i < len; i++) {
        const code = value.charCodeAt(i);
        if (code > 127)
          break;
        ascii[pos++] = code;
      }
      if (i == len) {
        this.pos = pos;
        return this;
      }
    }
    this.ensureCapacity(len * 3 + 5);
    const lenPrefixSizeGuess = varint32Size(len);
    const buf = this.buffer;
    const start3 = this.pos;
    const { written } = this.encodeUtf8Into(value, buf.subarray(start3 + lenPrefixSizeGuess));
    const lenPrefixSize = varint32Size(written);
    if (lenPrefixSize != lenPrefixSizeGuess) {
      buf.copyWithin(start3 + lenPrefixSize, start3 + lenPrefixSizeGuess, start3 + lenPrefixSizeGuess + written);
    }
    this.uint32(written);
    this.pos += written;
    return this;
  }
  /**
   * Write a `float` value, 32-bit floating point number.
   */
  float(value) {
    assertFloat32(value);
    this.ensureCapacity(4);
    this.view().setFloat32(this.pos, value, true);
    this.pos += 4;
    return this;
  }
  /**
   * Write a `double` value, a 64-bit floating point number.
   */
  double(value) {
    this.ensureCapacity(8);
    this.view().setFloat64(this.pos, value, true);
    this.pos += 8;
    return this;
  }
  /**
   * Write a `fixed32` value, an unsigned, fixed-length 32-bit integer.
   */
  fixed32(value) {
    assertUInt32(value);
    this.ensureCapacity(4);
    this.view().setUint32(this.pos, value, true);
    this.pos += 4;
    return this;
  }
  /**
   * Write a `sfixed32` value, a signed, fixed-length 32-bit integer.
   */
  sfixed32(value) {
    assertInt32(value);
    this.ensureCapacity(4);
    this.view().setInt32(this.pos, value, true);
    this.pos += 4;
    return this;
  }
  /**
   * Write a `sint32` value, a signed, zigzag-encoded 32-bit varint.
   */
  sint32(value) {
    assertInt32(value);
    return this.uint32((value << 1 ^ value >> 31) >>> 0);
  }
  /**
   * Write a `sfixed64` value, a signed, fixed-length 64-bit integer.
   */
  sfixed64(value) {
    const tc = protoInt64.enc(value);
    this.ensureCapacity(8);
    const view = this.view();
    view.setInt32(this.pos, tc.lo, true);
    view.setInt32(this.pos + 4, tc.hi, true);
    this.pos += 8;
    return this;
  }
  /**
   * Write a `fixed64` value, an unsigned, fixed-length 64 bit integer.
   */
  fixed64(value) {
    const tc = protoInt64.uEnc(value);
    this.ensureCapacity(8);
    const view = this.view();
    view.setInt32(this.pos, tc.lo, true);
    view.setInt32(this.pos + 4, tc.hi, true);
    this.pos += 8;
    return this;
  }
  /**
   * Write a `int64` value, a signed 64-bit varint.
   */
  int64(value) {
    const tc = protoInt64.enc(value);
    return this.writeVarint64(tc.lo, tc.hi);
  }
  /**
   * Write a `sint64` value, a signed, zig-zag-encoded 64-bit varint.
   */
  sint64(value) {
    const tc = protoInt64.enc(value), sign = tc.hi >> 31, lo = tc.lo << 1 ^ sign, hi = (tc.hi << 1 | tc.lo >>> 31) ^ sign;
    return this.writeVarint64(lo, hi);
  }
  /**
   * Write a `uint64` value, an unsigned 64-bit varint.
   */
  uint64(value) {
    const tc = protoInt64.uEnc(value);
    return this.writeVarint64(tc.lo, tc.hi);
  }
  /**
   * Write a 64-bit varint directly into the buffer. Accepts the value as
   * split low/high 32-bit words.
   *
   * Ported from varint64write() to avoid the intermediate number[] buffer.
   * See https://github.com/protocolbuffers/protobuf/blob/8a71927d74a4ce34efe2d8769fda198f52d20d12/js/experimental/runtime/kernel/writer.js#L344
   */
  writeVarint64(lo, hi) {
    this.ensureCapacity(10);
    const buf = this.buffer;
    let pos = this.pos;
    for (let i = 0; i < 28; i = i + 7) {
      const shift = lo >>> i;
      const hasNext = !(shift >>> 7 == 0 && hi == 0);
      buf[pos++] = (hasNext ? shift | 128 : shift) & 255;
      if (!hasNext) {
        this.pos = pos;
        return this;
      }
    }
    const splitBits = lo >>> 28 & 15 | (hi & 7) << 4;
    const hasMoreBits = !(hi >> 3 == 0);
    buf[pos++] = (hasMoreBits ? splitBits | 128 : splitBits) & 255;
    if (!hasMoreBits) {
      this.pos = pos;
      return this;
    }
    for (let i = 3; i < 31; i = i + 7) {
      const shift = hi >>> i;
      const hasNext = !(shift >>> 7 == 0);
      buf[pos++] = (hasNext ? shift | 128 : shift) & 255;
      if (!hasNext) {
        this.pos = pos;
        return this;
      }
    }
    buf[pos++] = hi >>> 31 & 1;
    this.pos = pos;
    return this;
  }
};
var INITIAL_SIZE = 128;
var DEFAULT_LEN_PREFIX_SIZE = 1;
var EMPTY_BUFFER = new Uint8Array(0);
var EMPTY_VIEW = new DataView(EMPTY_BUFFER.buffer);
var ASCII_MAX_LENGTH = 32;
function varint32Size(value) {
  if (value < 128)
    return 1;
  if (value < 16384)
    return 2;
  if (value < 2097152)
    return 3;
  if (value < 268435456)
    return 4;
  return 5;
}
var BinaryReader = class {
  constructor(buf, decodeUtf8 = getTextEncoding().decodeUtf8) {
    this.decodeUtf8 = decodeUtf8;
    this.varint64Lo = 0;
    this.varint64Hi = 0;
    this.varint64 = varint64read;
    this.uint32 = varint32read;
    this.buf = buf;
    this.len = buf.length;
    this.pos = 0;
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  /**
   * Reads a tag - field number and wire type. Tags are uint32 varints; values
   * that do not fit in uint32 are rejected.
   */
  tag() {
    const start3 = this.pos;
    const tag = this.uint32();
    const bytesRead = this.pos - start3;
    if (bytesRead > 5 || bytesRead == 5 && this.buf[this.pos - 1] > 15) {
      throw new Error("illegal tag: varint overflows uint32");
    }
    const fieldNo = tag >>> 3;
    const wireType = tag & 7;
    if (fieldNo <= 0 || wireType > 5) {
      throw new Error("illegal tag: field no " + fieldNo + " wire type " + wireType);
    }
    return [fieldNo, wireType];
  }
  /**
   * Skip one element and return the skipped data.
   *
   * When skipping StartGroup, provide the tags field number to check for
   * matching field number in the EndGroup tag. Recursion into nested groups
   * is guarded by the `recursionLimit` argument: When the limit is reached,
   * this method throws.
   */
  skip(wireType, fieldNo, recursionLimit = 100) {
    let start3 = this.pos;
    switch (wireType) {
      case WireType.Varint:
        while (this.buf[this.pos++] & 128) {
        }
        break;
      // @ts-ignore TS7029: Fallthrough case in switch -- ignore instead of expect-error for compiler settings without noFallthroughCasesInSwitch: true
      case WireType.Bit64:
        this.pos += 4;
      case WireType.Bit32:
        this.pos += 4;
        break;
      case WireType.LengthDelimited:
        let len = this.uint32();
        this.pos += len;
        break;
      case WireType.StartGroup:
        if (recursionLimit <= 0) {
          throw new Error("maximum recursion depth reached");
        }
        for (; ; ) {
          const [fn, wt] = this.tag();
          if (wt === WireType.EndGroup) {
            if (fieldNo !== void 0 && fn !== fieldNo) {
              throw new Error("invalid end group tag");
            }
            break;
          }
          this.skip(wt, fn, recursionLimit - 1);
        }
        break;
      default:
        throw new Error("cant skip wire type " + wireType);
    }
    this.assertBounds();
    return this.buf.subarray(start3, this.pos);
  }
  /**
   * Throws error if position in byte array is out of range.
   */
  assertBounds() {
    if (this.pos > this.len)
      throw new RangeError("premature EOF");
  }
  /**
   * Read a `int32` field, a signed 32 bit varint.
   */
  int32() {
    return this.uint32() | 0;
  }
  /**
   * Read a `sint32` field, a signed, zigzag-encoded 32-bit varint.
   */
  sint32() {
    let zze = this.uint32();
    return zze >>> 1 ^ -(zze & 1);
  }
  /**
   * Read a `int64` field, a signed 64-bit varint.
   */
  int64() {
    this.varint64();
    return protoInt64.dec(this.varint64Lo, this.varint64Hi);
  }
  /**
   * Read a `uint64` field, an unsigned 64-bit varint.
   */
  uint64() {
    this.varint64();
    return protoInt64.uDec(this.varint64Lo, this.varint64Hi);
  }
  /**
   * Read a `sint64` field, a signed, zig-zag-encoded 64-bit varint.
   */
  sint64() {
    this.varint64();
    let lo = this.varint64Lo;
    let hi = this.varint64Hi;
    let s = -(lo & 1);
    lo = (lo >>> 1 | (hi & 1) << 31) ^ s;
    hi = hi >>> 1 ^ s;
    return protoInt64.dec(lo, hi);
  }
  /**
   * Read a `bool` field, a variant.
   */
  bool() {
    const b = this.buf[this.pos];
    if (b < 128) {
      this.pos++;
      return b !== 0;
    }
    this.varint64();
    return this.varint64Lo !== 0 || this.varint64Hi !== 0;
  }
  /**
   * Read a `fixed32` field, an unsigned, fixed-length 32-bit integer.
   */
  fixed32() {
    return this.view.getUint32((this.pos += 4) - 4, true);
  }
  /**
   * Read a `sfixed32` field, a signed, fixed-length 32-bit integer.
   */
  sfixed32() {
    return this.view.getInt32((this.pos += 4) - 4, true);
  }
  /**
   * Read a `fixed64` field, an unsigned, fixed-length 64 bit integer.
   */
  fixed64() {
    return protoInt64.uDec(this.sfixed32(), this.sfixed32());
  }
  /**
   * Read a `fixed64` field, a signed, fixed-length 64-bit integer.
   */
  sfixed64() {
    return protoInt64.dec(this.sfixed32(), this.sfixed32());
  }
  /**
   * Read a `float` field, 32-bit floating point number.
   */
  float() {
    return this.view.getFloat32((this.pos += 4) - 4, true);
  }
  /**
   * Read a `double` field, a 64-bit floating point number.
   */
  double() {
    return this.view.getFloat64((this.pos += 8) - 8, true);
  }
  /**
   * Read a `bytes` field, length-delimited arbitrary data.
   */
  bytes() {
    let len = this.uint32(), start3 = this.pos;
    this.pos += len;
    this.assertBounds();
    return this.buf.subarray(start3, start3 + len);
  }
  /**
   * Read a `string` field, length-delimited data converted to UTF-8 text. If
   * `strict` is true, throw on invalid UTF-8 instead of substituting U+FFFD.
   */
  string(strict) {
    const bytes2 = this.bytes();
    const len = bytes2.length;
    if (len <= ASCII_MAX_LENGTH) {
      const codes = new Array(len);
      for (let i = 0; i < len; i++) {
        const byte = bytes2[i];
        if (byte > 127) {
          return this.decodeUtf8(bytes2, strict);
        }
        codes[i] = byte;
      }
      return String.fromCharCode.apply(String, codes);
    }
    return this.decodeUtf8(bytes2, strict);
  }
};
function assertInt32(arg) {
  if (typeof arg == "string") {
    arg = Number(arg);
  } else if (typeof arg != "number") {
    throw new Error("invalid int32: " + typeof arg);
  }
  if (!Number.isInteger(arg) || arg > INT32_MAX || arg < INT32_MIN)
    throw new Error("invalid int32: " + arg);
}
function assertUInt32(arg) {
  if (typeof arg == "string") {
    arg = Number(arg);
  } else if (typeof arg != "number") {
    throw new Error("invalid uint32: " + typeof arg);
  }
  if (!Number.isInteger(arg) || arg > UINT32_MAX || arg < 0)
    throw new Error("invalid uint32: " + arg);
}
function assertFloat32(arg) {
  if (typeof arg == "string") {
    const o = arg;
    arg = Number(arg);
    if (Number.isNaN(arg) && o !== "NaN") {
      throw new Error("invalid float32: " + o);
    }
  } else if (typeof arg != "number") {
    throw new Error("invalid float32: " + typeof arg);
  }
  if (!Number.isFinite(Math.fround(arg)) && Number.isFinite(arg)) {
    throw new Error("invalid float32: " + arg);
  }
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/registry.js
function createFileRegistry(...args) {
  const registry = createBaseRegistry();
  if (!args.length) {
    return registry;
  }
  if ("$typeName" in args[0] && args[0].$typeName == "google.protobuf.FileDescriptorSet") {
    for (const file of args[0].file) {
      addFile(file, registry);
    }
    return registry;
  }
  if ("$typeName" in args[0]) {
    let recurseDeps = function(file) {
      const deps = [];
      for (const protoFileName of file.dependency) {
        if (registry.getFile(protoFileName) != void 0) {
          continue;
        }
        if (seen.has(protoFileName)) {
          continue;
        }
        const dep = resolve(protoFileName);
        if (!dep) {
          throw new Error(`Unable to resolve ${protoFileName}, imported by ${file.name}`);
        }
        if ("kind" in dep) {
          registry.addFile(dep, false, true);
        } else {
          seen.add(dep.name);
          deps.push(dep);
        }
      }
      return deps.concat(...deps.map(recurseDeps));
    };
    const input2 = args[0];
    const resolve = args[1];
    const seen = /* @__PURE__ */ new Set();
    for (const file of [input2, ...recurseDeps(input2)].reverse()) {
      addFile(file, registry);
    }
  } else {
    for (const fileReg of args) {
      for (const file of fileReg.files) {
        registry.addFile(file);
      }
    }
  }
  return registry;
}
function createBaseRegistry() {
  const types = /* @__PURE__ */ new Map();
  const extendees = /* @__PURE__ */ new Map();
  const files = /* @__PURE__ */ new Map();
  return {
    kind: "registry",
    types,
    extendees,
    [Symbol.iterator]() {
      return types.values();
    },
    get files() {
      return files.values();
    },
    addFile(file, skipTypes, withDeps) {
      files.set(file.proto.name, file);
      if (!skipTypes) {
        for (const type of nestedTypes(file)) {
          this.add(type);
        }
      }
      if (withDeps) {
        for (const f of file.dependencies) {
          this.addFile(f, skipTypes, withDeps);
        }
      }
    },
    add(desc) {
      if (desc.kind == "extension") {
        let numberToExt = extendees.get(desc.extendee.typeName);
        if (!numberToExt) {
          extendees.set(
            desc.extendee.typeName,
            // biome-ignore lint/suspicious/noAssignInExpressions: no
            numberToExt = /* @__PURE__ */ new Map()
          );
        }
        numberToExt.set(desc.number, desc);
      }
      types.set(desc.typeName, desc);
    },
    get(typeName) {
      return types.get(typeName);
    },
    getFile(fileName) {
      return files.get(fileName);
    },
    getMessage(typeName) {
      const t = types.get(typeName);
      return (t === null || t === void 0 ? void 0 : t.kind) == "message" ? t : void 0;
    },
    getEnum(typeName) {
      const t = types.get(typeName);
      return (t === null || t === void 0 ? void 0 : t.kind) == "enum" ? t : void 0;
    },
    getExtension(typeName) {
      const t = types.get(typeName);
      return (t === null || t === void 0 ? void 0 : t.kind) == "extension" ? t : void 0;
    },
    getExtensionFor(extendee, no) {
      var _a;
      return (_a = extendees.get(extendee.typeName)) === null || _a === void 0 ? void 0 : _a.get(no);
    },
    getService(typeName) {
      const t = types.get(typeName);
      return (t === null || t === void 0 ? void 0 : t.kind) == "service" ? t : void 0;
    }
  };
}
var EDITION_PROTO2 = 998;
var EDITION_PROTO3 = 999;
var EDITION_UNSTABLE = 9999;
var TYPE_STRING = 9;
var TYPE_GROUP = 10;
var TYPE_MESSAGE = 11;
var TYPE_BYTES = 12;
var TYPE_ENUM = 14;
var LABEL_REPEATED = 3;
var LABEL_REQUIRED = 2;
var JS_STRING = 1;
var IDEMPOTENCY_UNKNOWN = 0;
var EXPLICIT = 1;
var IMPLICIT2 = 2;
var LEGACY_REQUIRED = 3;
var PACKED = 1;
var DELIMITED = 2;
var OPEN = 1;
var VERIFY = 2;
var maximumEdition = 1002;
var featureDefaults = {
  // EDITION_PROTO2
  998: {
    fieldPresence: 1,
    // EXPLICIT,
    enumType: 2,
    // CLOSED,
    repeatedFieldEncoding: 2,
    // EXPANDED,
    utf8Validation: 3,
    // NONE,
    messageEncoding: 1,
    // LENGTH_PREFIXED,
    jsonFormat: 2,
    // LEGACY_BEST_EFFORT,
    enforceNamingStyle: 2,
    // STYLE_LEGACY,
    defaultSymbolVisibility: 1,
    // EXPORT_ALL,
    enforceProtoLimits: 1
    // LEGACY_NO_EXPLICIT_LIMITS,
  },
  // EDITION_PROTO3
  999: {
    fieldPresence: 2,
    // IMPLICIT,
    enumType: 1,
    // OPEN,
    repeatedFieldEncoding: 1,
    // PACKED,
    utf8Validation: 2,
    // VERIFY,
    messageEncoding: 1,
    // LENGTH_PREFIXED,
    jsonFormat: 1,
    // ALLOW,
    enforceNamingStyle: 2,
    // STYLE_LEGACY,
    defaultSymbolVisibility: 1,
    // EXPORT_ALL,
    enforceProtoLimits: 1
    // LEGACY_NO_EXPLICIT_LIMITS,
  },
  // EDITION_2023
  1e3: {
    fieldPresence: 1,
    // EXPLICIT,
    enumType: 1,
    // OPEN,
    repeatedFieldEncoding: 1,
    // PACKED,
    utf8Validation: 2,
    // VERIFY,
    messageEncoding: 1,
    // LENGTH_PREFIXED,
    jsonFormat: 1,
    // ALLOW,
    enforceNamingStyle: 2,
    // STYLE_LEGACY,
    defaultSymbolVisibility: 1,
    // EXPORT_ALL,
    enforceProtoLimits: 1
    // LEGACY_NO_EXPLICIT_LIMITS,
  },
  // EDITION_2024
  1001: {
    fieldPresence: 1,
    // EXPLICIT,
    enumType: 1,
    // OPEN,
    repeatedFieldEncoding: 1,
    // PACKED,
    utf8Validation: 2,
    // VERIFY,
    messageEncoding: 1,
    // LENGTH_PREFIXED,
    jsonFormat: 1,
    // ALLOW,
    enforceNamingStyle: 1,
    // STYLE2024,
    defaultSymbolVisibility: 2,
    // EXPORT_TOP_LEVEL,
    enforceProtoLimits: 1
    // LEGACY_NO_EXPLICIT_LIMITS,
  },
  // EDITION_2026
  1002: {
    fieldPresence: 1,
    // EXPLICIT,
    enumType: 1,
    // OPEN,
    repeatedFieldEncoding: 1,
    // PACKED,
    utf8Validation: 2,
    // VERIFY,
    messageEncoding: 1,
    // LENGTH_PREFIXED,
    jsonFormat: 1,
    // ALLOW,
    enforceNamingStyle: 3,
    // STYLE2026,
    defaultSymbolVisibility: 4,
    // STRICT,
    enforceProtoLimits: 2
    // PROTO_LIMITS2026,
  }
};
function addFile(proto, reg) {
  var _a, _b;
  const file = {
    kind: "file",
    proto,
    deprecated: (_b = (_a = proto.options) === null || _a === void 0 ? void 0 : _a.deprecated) !== null && _b !== void 0 ? _b : false,
    edition: getFileEdition(proto),
    name: proto.name.replace(/\.proto$/, ""),
    dependencies: findFileDependencies(proto, reg),
    enums: [],
    messages: [],
    extensions: [],
    services: [],
    toString() {
      return `file ${proto.name}`;
    }
  };
  const mapEntriesStore = /* @__PURE__ */ new Map();
  const mapEntries = {
    get(typeName) {
      return mapEntriesStore.get(typeName);
    },
    add(desc) {
      var _a2;
      assert(((_a2 = desc.proto.options) === null || _a2 === void 0 ? void 0 : _a2.mapEntry) === true);
      mapEntriesStore.set(desc.typeName, desc);
    }
  };
  for (const enumProto of proto.enumType) {
    addEnum(enumProto, file, void 0, reg);
  }
  for (const messageProto of proto.messageType) {
    addMessage(messageProto, file, void 0, reg, mapEntries);
  }
  for (const serviceProto of proto.service) {
    addService(serviceProto, file, reg);
  }
  addExtensions(file, reg);
  for (const mapEntry of mapEntriesStore.values()) {
    addFields(mapEntry, reg, mapEntries);
  }
  for (const message of file.messages) {
    addFields(message, reg, mapEntries);
    addExtensions(message, reg);
  }
  reg.addFile(file, true);
}
function addExtensions(desc, reg) {
  switch (desc.kind) {
    case "file":
      for (const proto of desc.proto.extension) {
        const ext = newField(proto, desc, reg);
        desc.extensions.push(ext);
        reg.add(ext);
      }
      break;
    case "message":
      for (const proto of desc.proto.extension) {
        const ext = newField(proto, desc, reg);
        desc.nestedExtensions.push(ext);
        reg.add(ext);
      }
      for (const message of desc.nestedMessages) {
        addExtensions(message, reg);
      }
      break;
  }
}
function addFields(message, reg, mapEntries) {
  const allOneofs = message.proto.oneofDecl.map((proto) => newOneof(proto, message));
  const oneofsSeen = /* @__PURE__ */ new Set();
  for (const proto of message.proto.field) {
    const oneof = findOneof(proto, allOneofs);
    const field = newField(proto, message, reg, oneof, mapEntries);
    message.fields.push(field);
    message.field[field.localName] = field;
    if (oneof === void 0) {
      message.members.push(field);
    } else {
      oneof.fields.push(field);
      if (!oneofsSeen.has(oneof)) {
        oneofsSeen.add(oneof);
        message.members.push(oneof);
      }
    }
  }
  for (const oneof of allOneofs.filter((o) => oneofsSeen.has(o))) {
    message.oneofs.push(oneof);
  }
  for (const child of message.nestedMessages) {
    addFields(child, reg, mapEntries);
  }
}
function addEnum(proto, file, parent, reg) {
  var _a, _b, _c, _d, _e, _f;
  const sharedPrefix = findEnumSharedPrefix(proto.name, proto.value);
  const desc = {
    kind: "enum",
    proto,
    deprecated: (_b = (_a = proto.options) === null || _a === void 0 ? void 0 : _a.deprecated) !== null && _b !== void 0 ? _b : false,
    file,
    parent,
    open: true,
    name: proto.name,
    typeName: makeTypeName(proto, parent, file),
    value: {},
    values: [],
    sharedPrefix,
    toString() {
      return `enum ${this.typeName}`;
    }
  };
  desc.open = isEnumOpen(desc);
  reg.add(desc);
  for (const p of proto.value) {
    const name = p.name;
    desc.values.push(
      // biome-ignore lint/suspicious/noAssignInExpressions: no
      desc.value[p.number] = {
        kind: "enum_value",
        proto: p,
        deprecated: (_d = (_c = p.options) === null || _c === void 0 ? void 0 : _c.deprecated) !== null && _d !== void 0 ? _d : false,
        parent: desc,
        name,
        localName: safeObjectProperty(sharedPrefix == void 0 ? name : name.substring(sharedPrefix.length)),
        number: p.number,
        jsonName: (_e = findEnumValueJsonName(p)) !== null && _e !== void 0 ? _e : name,
        toString() {
          return `enum value ${desc.typeName}.${name}`;
        }
      }
    );
  }
  ((_f = parent === null || parent === void 0 ? void 0 : parent.nestedEnums) !== null && _f !== void 0 ? _f : file.enums).push(desc);
}
function addMessage(proto, file, parent, reg, mapEntries) {
  var _a, _b, _c, _d;
  const desc = {
    kind: "message",
    proto,
    deprecated: (_b = (_a = proto.options) === null || _a === void 0 ? void 0 : _a.deprecated) !== null && _b !== void 0 ? _b : false,
    file,
    parent,
    name: proto.name,
    typeName: makeTypeName(proto, parent, file),
    fields: [],
    field: {},
    oneofs: [],
    members: [],
    nestedEnums: [],
    nestedMessages: [],
    nestedExtensions: [],
    toString() {
      return `message ${this.typeName}`;
    }
  };
  if (((_c = proto.options) === null || _c === void 0 ? void 0 : _c.mapEntry) === true) {
    mapEntries.add(desc);
  } else {
    ((_d = parent === null || parent === void 0 ? void 0 : parent.nestedMessages) !== null && _d !== void 0 ? _d : file.messages).push(desc);
    reg.add(desc);
  }
  for (const enumProto of proto.enumType) {
    addEnum(enumProto, file, desc, reg);
  }
  for (const messageProto of proto.nestedType) {
    addMessage(messageProto, file, desc, reg, mapEntries);
  }
}
function addService(proto, file, reg) {
  var _a, _b;
  const desc = {
    kind: "service",
    proto,
    deprecated: (_b = (_a = proto.options) === null || _a === void 0 ? void 0 : _a.deprecated) !== null && _b !== void 0 ? _b : false,
    file,
    name: proto.name,
    typeName: makeTypeName(proto, void 0, file),
    methods: [],
    method: {},
    toString() {
      return `service ${this.typeName}`;
    }
  };
  file.services.push(desc);
  reg.add(desc);
  for (const methodProto of proto.method) {
    const method = newMethod(methodProto, desc, reg);
    desc.methods.push(method);
    desc.method[method.localName] = method;
  }
}
function newMethod(proto, parent, reg) {
  var _a, _b, _c, _d;
  let methodKind;
  if (proto.clientStreaming && proto.serverStreaming) {
    methodKind = "bidi_streaming";
  } else if (proto.clientStreaming) {
    methodKind = "client_streaming";
  } else if (proto.serverStreaming) {
    methodKind = "server_streaming";
  } else {
    methodKind = "unary";
  }
  const input2 = reg.getMessage(trimLeadingDot(proto.inputType));
  const output = reg.getMessage(trimLeadingDot(proto.outputType));
  assert(input2, `invalid MethodDescriptorProto: input_type ${proto.inputType} not found`);
  assert(output, `invalid MethodDescriptorProto: output_type ${proto.inputType} not found`);
  const name = proto.name;
  return {
    kind: "rpc",
    proto,
    deprecated: (_b = (_a = proto.options) === null || _a === void 0 ? void 0 : _a.deprecated) !== null && _b !== void 0 ? _b : false,
    parent,
    name,
    localName: safeObjectProperty(name.length ? safeObjectProperty(name[0].toLowerCase() + name.substring(1)) : name),
    methodKind,
    input: input2,
    output,
    idempotency: (_d = (_c = proto.options) === null || _c === void 0 ? void 0 : _c.idempotencyLevel) !== null && _d !== void 0 ? _d : IDEMPOTENCY_UNKNOWN,
    toString() {
      return `rpc ${parent.typeName}.${name}`;
    }
  };
}
function newOneof(proto, parent) {
  return {
    kind: "oneof",
    proto,
    deprecated: false,
    parent,
    fields: [],
    name: proto.name,
    localName: safeObjectProperty(protoCamelCase(proto.name)),
    toString() {
      return `oneof ${parent.typeName}.${this.name}`;
    }
  };
}
function newField(proto, parentOrFile, reg, oneof, mapEntries) {
  var _a, _b, _c;
  const isExtension = mapEntries === void 0;
  const field = {
    kind: "field",
    proto,
    deprecated: (_b = (_a = proto.options) === null || _a === void 0 ? void 0 : _a.deprecated) !== null && _b !== void 0 ? _b : false,
    name: proto.name,
    number: proto.number,
    scalar: void 0,
    message: void 0,
    enum: void 0,
    presence: getFieldPresence(proto, oneof, isExtension, parentOrFile),
    utf8Validation: isUtf8Validated(proto, parentOrFile),
    listKind: void 0,
    mapKind: void 0,
    mapKey: void 0,
    delimitedEncoding: void 0,
    packed: void 0,
    longAsString: false,
    getDefaultValue: void 0
  };
  let toStr;
  if (isExtension) {
    const file = parentOrFile.kind == "file" ? parentOrFile : parentOrFile.file;
    const parent = parentOrFile.kind == "file" ? void 0 : parentOrFile;
    const typeName = makeTypeName(proto, parent, file);
    field.kind = "extension";
    field.file = file;
    field.parent = parent;
    field.oneof = void 0;
    field.typeName = typeName;
    field.jsonName = `[${typeName}]`;
    toStr = () => `extension ${typeName}`;
    const extendee = reg.getMessage(trimLeadingDot(proto.extendee));
    assert(extendee, `invalid FieldDescriptorProto: extendee ${proto.extendee} not found`);
    field.extendee = extendee;
  } else {
    const parent = parentOrFile;
    assert(parent.kind == "message");
    field.parent = parent;
    field.oneof = oneof;
    field.localName = oneof ? protoCamelCase(proto.name) : safeObjectProperty(protoCamelCase(proto.name));
    field.jsonName = proto.jsonName;
    toStr = () => `field ${parent.typeName}.${proto.name}`;
  }
  Object.defineProperty(field, "toString", {
    value: toStr,
    writable: true,
    enumerable: true,
    configurable: true
  });
  const label = proto.label;
  const type = proto.type;
  const jstype = (_c = proto.options) === null || _c === void 0 ? void 0 : _c.jstype;
  if (label === LABEL_REPEATED) {
    const mapEntry = type == TYPE_MESSAGE ? mapEntries === null || mapEntries === void 0 ? void 0 : mapEntries.get(trimLeadingDot(proto.typeName)) : void 0;
    if (mapEntry) {
      field.fieldKind = "map";
      const { key, value } = findMapEntryFields(mapEntry);
      field.mapKey = key.scalar;
      field.mapKind = value.fieldKind;
      field.message = value.message;
      field.delimitedEncoding = false;
      field.enum = value.enum;
      field.scalar = value.scalar;
      return field;
    }
    field.fieldKind = "list";
    switch (type) {
      case TYPE_MESSAGE:
      case TYPE_GROUP:
        field.listKind = "message";
        field.message = reg.getMessage(trimLeadingDot(proto.typeName));
        assert(field.message);
        field.delimitedEncoding = isDelimitedEncoding(proto, parentOrFile);
        break;
      case TYPE_ENUM:
        field.listKind = "enum";
        field.enum = reg.getEnum(trimLeadingDot(proto.typeName));
        assert(field.enum);
        break;
      default:
        field.listKind = "scalar";
        field.scalar = type;
        field.longAsString = jstype == JS_STRING;
        break;
    }
    field.packed = isPackedField(proto, parentOrFile);
    return field;
  }
  switch (type) {
    case TYPE_MESSAGE:
    case TYPE_GROUP:
      field.fieldKind = "message";
      field.message = reg.getMessage(trimLeadingDot(proto.typeName));
      assert(field.message, `invalid FieldDescriptorProto: type_name ${proto.typeName} not found`);
      field.delimitedEncoding = isDelimitedEncoding(proto, parentOrFile);
      field.getDefaultValue = () => void 0;
      break;
    case TYPE_ENUM: {
      const enumeration = reg.getEnum(trimLeadingDot(proto.typeName));
      assert(enumeration !== void 0, `invalid FieldDescriptorProto: type_name ${proto.typeName} not found`);
      field.fieldKind = "enum";
      field.enum = reg.getEnum(trimLeadingDot(proto.typeName));
      field.getDefaultValue = () => {
        return unsafeIsSetExplicit(proto, "defaultValue") ? parseTextFormatEnumValue(enumeration, proto.defaultValue) : void 0;
      };
      break;
    }
    default: {
      field.fieldKind = "scalar";
      field.scalar = type;
      field.longAsString = jstype == JS_STRING;
      field.getDefaultValue = () => {
        return unsafeIsSetExplicit(proto, "defaultValue") ? parseTextFormatScalarValue(type, proto.defaultValue) : void 0;
      };
      break;
    }
  }
  return field;
}
function getFileEdition(proto) {
  switch (proto.syntax) {
    case "":
    case "proto2":
      return EDITION_PROTO2;
    case "proto3":
      return EDITION_PROTO3;
    case "editions":
      if (proto.edition === EDITION_UNSTABLE) {
        return maximumEdition;
      }
      if (proto.edition in featureDefaults) {
        return proto.edition;
      }
      throw new Error(`${proto.name}: unsupported edition`);
    default:
      throw new Error(`${proto.name}: unsupported syntax "${proto.syntax}"`);
  }
}
function findFileDependencies(proto, reg) {
  return proto.dependency.map((wantName) => {
    const dep = reg.getFile(wantName);
    if (!dep) {
      throw new Error(`Cannot find ${wantName}, imported by ${proto.name}`);
    }
    return dep;
  });
}
function findEnumSharedPrefix(enumName, values) {
  const prefix2 = camelToSnakeCase(enumName) + "_";
  for (const value of values) {
    if (!value.name.toLowerCase().startsWith(prefix2)) {
      return void 0;
    }
    const shortName = value.name.substring(prefix2.length);
    if (shortName.length == 0) {
      return void 0;
    }
    if (/^\d/.test(shortName)) {
      return void 0;
    }
  }
  return prefix2;
}
function findEnumValueJsonName(proto) {
  var _a, _b;
  const extensionNumber = 998;
  const fieldNumber = 1;
  let jsonName;
  for (const uf of (_b = (_a = proto.options) === null || _a === void 0 ? void 0 : _a.$unknown) !== null && _b !== void 0 ? _b : []) {
    if (uf.no !== extensionNumber || uf.wireType !== WireType.LengthDelimited) {
      continue;
    }
    const reader = new BinaryReader(new BinaryReader(uf.data).bytes());
    while (reader.pos < reader.len) {
      const [no, wireType] = reader.tag();
      if (no === fieldNumber && wireType === WireType.LengthDelimited) {
        jsonName = reader.string();
      } else {
        reader.skip(wireType, no);
      }
    }
  }
  return jsonName;
}
function camelToSnakeCase(camel) {
  return (camel.substring(0, 1) + camel.substring(1).replace(/[A-Z]/g, (c) => "_" + c)).toLowerCase();
}
function makeTypeName(proto, parent, file) {
  let typeName;
  if (parent) {
    typeName = `${parent.typeName}.${proto.name}`;
  } else if (file.proto.package.length > 0) {
    typeName = `${file.proto.package}.${proto.name}`;
  } else {
    typeName = `${proto.name}`;
  }
  return typeName;
}
function trimLeadingDot(typeName) {
  return typeName.startsWith(".") ? typeName.substring(1) : typeName;
}
function findOneof(proto, allOneofs) {
  if (!unsafeIsSetExplicit(proto, "oneofIndex")) {
    return void 0;
  }
  if (proto.proto3Optional) {
    return void 0;
  }
  const oneof = allOneofs[proto.oneofIndex];
  assert(oneof, `invalid FieldDescriptorProto: oneof #${proto.oneofIndex} for field #${proto.number} not found`);
  return oneof;
}
function getFieldPresence(proto, oneof, isExtension, parent) {
  if (proto.label == LABEL_REQUIRED) {
    return LEGACY_REQUIRED;
  }
  if (proto.label == LABEL_REPEATED) {
    return IMPLICIT2;
  }
  if (!!oneof || proto.proto3Optional) {
    return EXPLICIT;
  }
  if (isExtension) {
    return EXPLICIT;
  }
  const resolved = resolveFeature("fieldPresence", { proto, parent });
  if (resolved == IMPLICIT2 && (proto.type == TYPE_MESSAGE || proto.type == TYPE_GROUP)) {
    return EXPLICIT;
  }
  return resolved;
}
function isPackedField(proto, parent) {
  if (proto.label != LABEL_REPEATED) {
    return false;
  }
  switch (proto.type) {
    case TYPE_STRING:
    case TYPE_BYTES:
    case TYPE_GROUP:
    case TYPE_MESSAGE:
      return false;
  }
  const o = proto.options;
  if (o && unsafeIsSetExplicit(o, "packed")) {
    return o.packed;
  }
  return PACKED == resolveFeature("repeatedFieldEncoding", {
    proto,
    parent
  });
}
function findMapEntryFields(mapEntry) {
  const key = mapEntry.fields.find((f) => f.number === 1);
  const value = mapEntry.fields.find((f) => f.number === 2);
  assert(key && key.fieldKind == "scalar" && key.scalar != ScalarType.BYTES && key.scalar != ScalarType.FLOAT && key.scalar != ScalarType.DOUBLE && value && value.fieldKind != "list" && value.fieldKind != "map");
  return { key, value };
}
function isEnumOpen(desc) {
  var _a;
  return OPEN == resolveFeature("enumType", {
    proto: desc.proto,
    parent: (_a = desc.parent) !== null && _a !== void 0 ? _a : desc.file
  });
}
function isDelimitedEncoding(proto, parent) {
  if (proto.type == TYPE_GROUP) {
    return true;
  }
  return DELIMITED == resolveFeature("messageEncoding", {
    proto,
    parent
  });
}
function isUtf8Validated(proto, parent) {
  return VERIFY == resolveFeature("utf8Validation", {
    proto,
    parent
  });
}
function resolveFeature(name, ref) {
  var _a, _b;
  const featureSet = (_a = ref.proto.options) === null || _a === void 0 ? void 0 : _a.features;
  if (featureSet) {
    const val = featureSet[name];
    if (val != 0) {
      return val;
    }
  }
  if ("kind" in ref) {
    if (ref.kind == "message") {
      return resolveFeature(name, (_b = ref.parent) !== null && _b !== void 0 ? _b : ref.file);
    }
    const editionDefaults = featureDefaults[ref.edition];
    if (!editionDefaults) {
      throw new Error(`feature default for edition ${ref.edition} not found`);
    }
    return editionDefaults[name];
  }
  return resolveFeature(name, ref.parent);
}
function assert(condition, msg) {
  if (!condition) {
    throw new Error(msg);
  }
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/codegenv2/boot.js
function boot(boot2) {
  const root = bootFileDescriptorProto(boot2);
  root.messageType.forEach(restoreJsonNames);
  const reg = createFileRegistry(root, () => void 0);
  return reg.getFile(root.name);
}
function bootFileDescriptorProto(init6) {
  const proto = /* @__PURE__ */ Object.create({
    syntax: "",
    edition: 0
  });
  return Object.assign(proto, Object.assign(Object.assign({ $typeName: "google.protobuf.FileDescriptorProto", dependency: [], publicDependency: [], weakDependency: [], optionDependency: [], service: [], extension: [] }, init6), { messageType: init6.messageType.map(bootDescriptorProto), enumType: init6.enumType.map(bootEnumDescriptorProto) }));
}
function bootDescriptorProto(init6) {
  var _a, _b, _c, _d, _e, _f, _g, _h;
  const proto = /* @__PURE__ */ Object.create({
    visibility: 0
  });
  return Object.assign(proto, {
    $typeName: "google.protobuf.DescriptorProto",
    name: init6.name,
    field: (_b = (_a = init6.field) === null || _a === void 0 ? void 0 : _a.map(bootFieldDescriptorProto)) !== null && _b !== void 0 ? _b : [],
    extension: [],
    nestedType: (_d = (_c = init6.nestedType) === null || _c === void 0 ? void 0 : _c.map(bootDescriptorProto)) !== null && _d !== void 0 ? _d : [],
    enumType: (_f = (_e = init6.enumType) === null || _e === void 0 ? void 0 : _e.map(bootEnumDescriptorProto)) !== null && _f !== void 0 ? _f : [],
    extensionRange: (_h = (_g = init6.extensionRange) === null || _g === void 0 ? void 0 : _g.map((e) => Object.assign({ $typeName: "google.protobuf.DescriptorProto.ExtensionRange" }, e))) !== null && _h !== void 0 ? _h : [],
    oneofDecl: [],
    reservedRange: [],
    reservedName: []
  });
}
function bootFieldDescriptorProto(init6) {
  const proto = /* @__PURE__ */ Object.create({
    label: 1,
    typeName: "",
    extendee: "",
    defaultValue: "",
    oneofIndex: 0,
    jsonName: "",
    proto3Optional: false
  });
  return Object.assign(proto, Object.assign(Object.assign({ $typeName: "google.protobuf.FieldDescriptorProto" }, init6), { options: init6.options ? bootFieldOptions(init6.options) : void 0 }));
}
function bootFieldOptions(init6) {
  var _a, _b, _c;
  const proto = /* @__PURE__ */ Object.create({
    ctype: 0,
    packed: false,
    jstype: 0,
    lazy: false,
    unverifiedLazy: false,
    deprecated: false,
    weak: false,
    debugRedact: false,
    retention: 0
  });
  return Object.assign(proto, Object.assign(Object.assign({ $typeName: "google.protobuf.FieldOptions" }, init6), { targets: (_a = init6.targets) !== null && _a !== void 0 ? _a : [], editionDefaults: (_c = (_b = init6.editionDefaults) === null || _b === void 0 ? void 0 : _b.map((e) => Object.assign({ $typeName: "google.protobuf.FieldOptions.EditionDefault" }, e))) !== null && _c !== void 0 ? _c : [], uninterpretedOption: [] }));
}
function bootEnumDescriptorProto(init6) {
  const proto = /* @__PURE__ */ Object.create({
    visibility: 0
  });
  return Object.assign(proto, {
    $typeName: "google.protobuf.EnumDescriptorProto",
    name: init6.name,
    reservedName: [],
    reservedRange: [],
    value: init6.value.map((e) => Object.assign({ $typeName: "google.protobuf.EnumValueDescriptorProto" }, e))
  });
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/wire/base64-encoding.js
var nativeSetFromBase64 = Uint8Array.prototype.setFromBase64;
function base64Decode(base64Str) {
  const len = base64Str.length;
  let size2 = len - (len + 3 >> 2);
  if ((len & 3) == 0 && base64Str[len - 1] == "=") {
    size2 -= base64Str[len - 2] == "=" ? 2 : 1;
  }
  const bytes2 = new Uint8Array(size2);
  let written = -1;
  if (nativeSetFromBase64) {
    try {
      const result = nativeSetFromBase64.call(bytes2, base64Str);
      if (result.read == len) {
        written = result.written;
      }
    } catch (_a) {
    }
  }
  if (written < 0) {
    written = setFromBase64(bytes2, base64Str);
  }
  return written == size2 ? bytes2 : bytes2.subarray(0, written);
}
function setFromBase64(bytes2, base64Str) {
  const table2 = getDecodeTable();
  let bytePos = 0, groupPos = 0, b, p = 0;
  for (let i = 0; i < base64Str.length; i++) {
    b = table2[base64Str.charCodeAt(i)];
    if (b === void 0) {
      switch (base64Str[i]) {
        // @ts-ignore TS7029: Fallthrough case in switch -- ignore instead of expect-error for compiler settings without noFallthroughCasesInSwitch: true
        case "=":
          groupPos = 0;
        // reset state when padding found
        case "\n":
        case "\r":
        case "	":
        case " ":
          continue;
        // skip white-space, and padding
        default:
          throw Error("invalid base64 string");
      }
    }
    switch (groupPos) {
      case 0:
        p = b;
        groupPos = 1;
        break;
      case 1:
        bytes2[bytePos++] = p << 2 | (b & 48) >> 4;
        p = b;
        groupPos = 2;
        break;
      case 2:
        bytes2[bytePos++] = (p & 15) << 4 | (b & 60) >> 2;
        p = b;
        groupPos = 3;
        break;
      case 3:
        bytes2[bytePos++] = (p & 3) << 6 | b;
        groupPos = 0;
        break;
    }
  }
  if (groupPos == 1)
    throw Error("invalid base64 string");
  return bytePos;
}
var nativeToBase64 = Uint8Array.prototype.toBase64;
var toBase64OptionsMap = {
  std: { alphabet: "base64", omitPadding: false },
  std_raw: { alphabet: "base64", omitPadding: true },
  url: { alphabet: "base64url", omitPadding: true }
};
function base64Encode(bytes2, encoding = "std") {
  if (nativeToBase64) {
    return nativeToBase64.call(bytes2, toBase64OptionsMap[encoding]);
  }
  const table2 = getEncodeTable(encoding);
  const pad = encoding == "std";
  let base64 = "", groupPos = 0, b, p = 0;
  for (let i = 0; i < bytes2.length; i++) {
    b = bytes2[i];
    switch (groupPos) {
      case 0:
        base64 += table2[b >> 2];
        p = (b & 3) << 4;
        groupPos = 1;
        break;
      case 1:
        base64 += table2[p | b >> 4];
        p = (b & 15) << 2;
        groupPos = 2;
        break;
      case 2:
        base64 += table2[p | b >> 6];
        base64 += table2[b & 63];
        groupPos = 0;
        break;
    }
  }
  if (groupPos) {
    base64 += table2[p];
    if (pad) {
      base64 += "=";
      if (groupPos == 1)
        base64 += "=";
    }
  }
  return base64;
}
var encodeTableStd;
var encodeTableUrl;
var decodeTable;
function getEncodeTable(encoding) {
  if (!encodeTableStd) {
    encodeTableStd = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".split("");
    encodeTableUrl = encodeTableStd.slice(0, -2).concat("-", "_");
  }
  return encoding == "url" ? (
    // biome-ignore lint/style/noNonNullAssertion: TS fails to narrow down
    encodeTableUrl
  ) : encodeTableStd;
}
function getDecodeTable() {
  if (!decodeTable) {
    decodeTable = [];
    const encodeTable = getEncodeTable("std");
    for (let i = 0; i < encodeTable.length; i++)
      decodeTable[encodeTable[i].charCodeAt(0)] = i;
    decodeTable["-".charCodeAt(0)] = encodeTable.indexOf("+");
    decodeTable["_".charCodeAt(0)] = encodeTable.indexOf("/");
  }
  return decodeTable;
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/reflect/error.js
var errorNames = [
  "FieldValueInvalidError",
  "FieldListRangeError",
  "ForeignFieldError"
];
var FieldError = class extends Error {
  constructor(fieldOrOneof, message, name = "FieldValueInvalidError") {
    super(message);
    this.name = name;
    this.field = () => fieldOrOneof;
  }
};
function isFieldError(arg) {
  return arg instanceof Error && errorNames.includes(arg.name) && "field" in arg && typeof arg.field == "function";
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/is-message.js
function isMessage(arg, schema) {
  const isMessage2 = arg !== null && typeof arg == "object" && "$typeName" in arg && typeof arg.$typeName == "string";
  if (!isMessage2) {
    return false;
  }
  if (schema === void 0) {
    return true;
  }
  return schema.typeName === arg.$typeName;
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/reflect/guard.js
function isObject(arg) {
  return arg !== null && typeof arg == "object" && !Array.isArray(arg);
}
function isReflectList(arg, field) {
  var _a, _b, _c, _d;
  if (isObject(arg) && unsafeLocal in arg && "add" in arg && "field" in arg && typeof arg.field == "function") {
    if (field !== void 0) {
      const a = field;
      const b = arg.field();
      return a.listKind == b.listKind && a.scalar === b.scalar && ((_a = a.message) === null || _a === void 0 ? void 0 : _a.typeName) === ((_b = b.message) === null || _b === void 0 ? void 0 : _b.typeName) && ((_c = a.enum) === null || _c === void 0 ? void 0 : _c.typeName) === ((_d = b.enum) === null || _d === void 0 ? void 0 : _d.typeName);
    }
    return true;
  }
  return false;
}
function isReflectMap(arg, field) {
  var _a, _b, _c, _d;
  if (isObject(arg) && unsafeLocal in arg && "has" in arg && "field" in arg && typeof arg.field == "function") {
    if (field !== void 0) {
      const a = field, b = arg.field();
      return a.mapKey === b.mapKey && a.mapKind == b.mapKind && a.scalar === b.scalar && ((_a = a.message) === null || _a === void 0 ? void 0 : _a.typeName) === ((_b = b.message) === null || _b === void 0 ? void 0 : _b.typeName) && ((_c = a.enum) === null || _c === void 0 ? void 0 : _c.typeName) === ((_d = b.enum) === null || _d === void 0 ? void 0 : _d.typeName);
    }
    return true;
  }
  return false;
}
function isReflectMessage(arg, messageDesc2) {
  return isObject(arg) && unsafeLocal in arg && "desc" in arg && isObject(arg.desc) && arg.desc.kind === "message" && (messageDesc2 === void 0 || arg.desc.typeName == messageDesc2.typeName);
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/wkt/wrappers.js
function isWrapper(arg) {
  return isWrapperTypeName(arg.$typeName);
}
function isWrapperDesc(messageDesc2) {
  const f = messageDesc2.fields[0];
  return isWrapperTypeName(messageDesc2.typeName) && f !== void 0 && f.fieldKind == "scalar" && f.name == "value" && f.number == 1;
}
function hasCustomJsonRepresentation(desc) {
  switch (desc.typeName) {
    case "google.protobuf.Any":
    case "google.protobuf.Timestamp":
    case "google.protobuf.Duration":
    case "google.protobuf.FieldMask":
    case "google.protobuf.Struct":
    case "google.protobuf.Value":
    case "google.protobuf.ListValue":
      return true;
    default:
      return isWrapperDesc(desc);
  }
}
var wrapperTypeNames = /* @__PURE__ */ new Set([
  "google.protobuf.DoubleValue",
  "google.protobuf.FloatValue",
  "google.protobuf.Int64Value",
  "google.protobuf.UInt64Value",
  "google.protobuf.Int32Value",
  "google.protobuf.UInt32Value",
  "google.protobuf.BoolValue",
  "google.protobuf.StringValue",
  "google.protobuf.BytesValue"
]);
function isWrapperTypeName(name) {
  return wrapperTypeNames.has(name);
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/create.js
var EDITION_PROTO32 = 999;
var EDITION_PROTO22 = 998;
var IMPLICIT3 = 2;
function create(schema, init6) {
  if (isMessage(init6, schema)) {
    return init6;
  }
  return compiledCreate(schema)(init6);
}
var compiledCreates = /* @__PURE__ */ new WeakMap();
function compiledCreate(desc) {
  let compiled = compiledCreates.get(desc);
  if (compiled === void 0) {
    compiled = compileCreate(desc);
    compiledCreates.set(desc, compiled);
  }
  return compiled;
}
var INIT_SINGULAR = 0;
var INIT_LIST = 1;
var INIT_MAP = 2;
var INIT_ONEOF = 3;
function compileCreate(desc) {
  const typeName = desc.typeName;
  const { properties, prototype } = compileInitMessage(desc);
  return (init6) => {
    let message;
    if (prototype !== void 0) {
      message = Object.create(prototype);
      message.$typeName = typeName;
    } else {
      message = { $typeName: typeName };
    }
    for (let i = 0; i < properties.length; i++) {
      const property = properties[i];
      const name = property.name;
      const initValue = init6 === null || init6 === void 0 ? void 0 : init6[name];
      switch (property.kind) {
        case INIT_SINGULAR:
          if (initValue != null) {
            message[name] = property.convert !== void 0 ? property.convert(initValue) : initValue;
          } else if (property.constant !== void 0) {
            message[name] = property.constant;
          }
          break;
        case INIT_LIST:
          message[name] = property.convert !== void 0 && Array.isArray(initValue) ? initValue.map(property.convert) : initValue !== null && initValue !== void 0 ? initValue : [];
          break;
        case INIT_MAP:
          if (property.convert === void 0 || !isObject(initValue)) {
            message[name] = initValue !== null && initValue !== void 0 ? initValue : {};
          } else {
            const converted = {};
            const keys = Object.keys(initValue);
            for (let k = 0; k < keys.length; k++) {
              setOwn(converted, keys[k], property.convert(initValue[keys[k]]));
            }
            message[name] = converted;
          }
          break;
        case INIT_ONEOF: {
          const oneofValue = initValue;
          if ((oneofValue === null || oneofValue === void 0 ? void 0 : oneofValue.case) != null) {
            const convert = property.convert.get(oneofValue.case);
            if (convert !== void 0) {
              message[name] = {
                case: oneofValue.case,
                value: convert(oneofValue.value)
              };
              break;
            }
          }
          message[name] = { case: void 0 };
          break;
        }
      }
    }
    return message;
  };
}
function compileInitMessage(desc) {
  var _a, _b;
  const properties = [];
  const prototype = {};
  const usePrototype = needsPrototypeChain(desc);
  for (const member of desc.members) {
    const name = member.localName;
    if (member.kind == "oneof") {
      properties.push({
        name,
        kind: INIT_ONEOF,
        constant: void 0,
        convert: compileConvertOneof(member)
      });
      continue;
    }
    switch (member.fieldKind) {
      case "message": {
        properties.push({
          name,
          kind: INIT_SINGULAR,
          constant: void 0,
          convert: compileConvertMessage(member)
        });
        break;
      }
      case "list": {
        properties.push({
          name,
          kind: INIT_LIST,
          constant: void 0,
          convert: member.listKind == "message" ? (_a = compileConvertMessage(member)) !== null && _a !== void 0 ? _a : ((value) => value) : member.scalar == ScalarType.BYTES ? toU8Arr : void 0
        });
        break;
      }
      case "map": {
        properties.push({
          name,
          kind: INIT_MAP,
          constant: void 0,
          convert: member.mapKind == "message" ? (_b = compileConvertMessage(member)) !== null && _b !== void 0 ? _b : ((value) => value) : member.scalar == ScalarType.BYTES ? toU8Arr : void 0
        });
        break;
      }
      default: {
        const zeroValue = createZeroValue(member);
        properties.push({
          name,
          kind: INIT_SINGULAR,
          constant: member.presence == IMPLICIT3 ? zeroValue : void 0,
          convert: member.fieldKind == "scalar" && member.scalar == ScalarType.BYTES ? toU8Arr : void 0
        });
        if (usePrototype) {
          prototype[name] = zeroValue;
        }
        break;
      }
    }
  }
  return {
    properties,
    prototype: usePrototype ? prototype : void 0
  };
}
function compileConvertOneof(oneof) {
  const converters = /* @__PURE__ */ new Map();
  for (const field of oneof.fields) {
    let convert;
    if (field.fieldKind == "message") {
      convert = compileConvertMessage(field);
    } else if (field.fieldKind == "scalar" && field.scalar == ScalarType.BYTES) {
      convert = toU8Arr;
    }
    converters.set(field.localName, convert !== null && convert !== void 0 ? convert : ((value) => value));
  }
  return converters;
}
function compileConvertMessage(field) {
  if (field.fieldKind == "message" && !field.oneof && isWrapperDesc(field.message)) {
    return field.message.fields[0].scalar == ScalarType.BYTES ? toU8Arr : void 0;
  }
  if (field.message.typeName == "google.protobuf.Struct" && field.parent.typeName !== "google.protobuf.Value") {
    return void 0;
  }
  const messageDesc2 = field.message;
  let compiled;
  return (value) => {
    if (!isObject(value) || isMessage(value, messageDesc2)) {
      return value;
    }
    compiled !== null && compiled !== void 0 ? compiled : compiled = compiledCreate(messageDesc2);
    return compiled(value);
  };
}
function toU8Arr(value) {
  return Array.isArray(value) ? new Uint8Array(value) : value;
}
function needsPrototypeChain(desc) {
  switch (desc.file.edition) {
    case EDITION_PROTO32:
      return false;
    case EDITION_PROTO22:
      return true;
    default:
      return desc.fields.some((f) => f.presence != IMPLICIT3 && f.fieldKind != "message" && !f.oneof);
  }
}
function createZeroValue(field) {
  const defaultValue = field.getDefaultValue();
  if (defaultValue !== void 0) {
    return field.fieldKind == "scalar" && field.longAsString ? defaultValue.toString() : defaultValue;
  }
  return field.fieldKind == "scalar" ? scalarZeroValue(field.scalar, field.longAsString) : field.enum.values[0].number;
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/reflect/message.js
var NULL_VALUE = 0;
function localMessageMapper(field) {
  if (usesJsonRepresentation(field)) {
    return {
      toMessage: (local) => wktStructToReflect(local),
      toLocal: (message) => wktStructToLocal(message)
    };
  }
  if (field.fieldKind == "message" && !field.oneof && isWrapperDesc(field.message)) {
    const wrapperDesc = field.message;
    const valueLocalName = wrapperDesc.fields[0].localName;
    return {
      toMessage: (local) => {
        const message = create(wrapperDesc);
        if (local !== void 0) {
          message[valueLocalName] = local;
        }
        return message;
      },
      toLocal: (message) => message[valueLocalName]
    };
  }
  const childDesc = field.message;
  return {
    toMessage: (local) => local === void 0 ? create(childDesc) : local,
    toLocal: (message) => message
  };
}
function usesJsonRepresentation(field) {
  return field.message.typeName == "google.protobuf.Struct" && field.parent.typeName != "google.protobuf.Value";
}
function wktStructToReflect(json) {
  const struct = {
    $typeName: "google.protobuf.Struct",
    fields: {}
  };
  if (isObject(json)) {
    for (const k of Object.keys(json)) {
      setOwn(struct.fields, k, wktValueToReflect(json[k]));
    }
  }
  return struct;
}
function wktStructToLocal(val) {
  const json = {};
  for (const k of Object.keys(val.fields)) {
    setOwn(json, k, wktValueToLocal(val.fields[k]));
  }
  return json;
}
function wktValueToLocal(val) {
  switch (val.kind.case) {
    case "structValue":
      return wktStructToLocal(val.kind.value);
    case "listValue":
      return val.kind.value.values.map(wktValueToLocal);
    case "nullValue":
    case void 0:
      return null;
    default:
      return val.kind.value;
  }
}
function wktValueToReflect(json) {
  const value = {
    $typeName: "google.protobuf.Value",
    kind: { case: void 0 }
  };
  switch (typeof json) {
    case "number":
      value.kind = { case: "numberValue", value: json };
      break;
    case "string":
      value.kind = { case: "stringValue", value: json };
      break;
    case "boolean":
      value.kind = { case: "boolValue", value: json };
      break;
    case "object":
      if (json === null) {
        value.kind = { case: "nullValue", value: NULL_VALUE };
      } else if (Array.isArray(json)) {
        const listValue = {
          $typeName: "google.protobuf.ListValue",
          values: []
        };
        if (Array.isArray(json)) {
          for (const e of json) {
            listValue.values.push(wktValueToReflect(e));
          }
        }
        value.kind = {
          case: "listValue",
          value: listValue
        };
      } else {
        value.kind = {
          case: "structValue",
          value: wktStructToReflect(json)
        };
      }
      break;
  }
  return value;
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/to-binary.js
var IMPLICIT4 = 2;
var LEGACY_REQUIRED2 = 3;
var writeDefaults = {
  writeUnknownFields: true
};
function makeWriteOptions(options2) {
  return options2 ? Object.assign(Object.assign({}, writeDefaults), options2) : writeDefaults;
}
function toBinary(schema, message, options2) {
  const writer = new BinaryWriter();
  compiledWriter(schema)(writer, makeWriteOptions(options2), message);
  return writer.finish();
}
var compiledWriters = /* @__PURE__ */ new WeakMap();
function compiledWriter(desc) {
  let compiled = compiledWriters.get(desc);
  if (compiled === void 0) {
    compiled = compileMessage(desc);
  }
  return compiled;
}
function compileMessage(desc) {
  const typeName = desc.typeName;
  const sortedFields = desc.fields.concat().sort((a, b) => a.number - b.number);
  const foreignField = sortedFields[0];
  const fieldWriters = [];
  const compiled = (writer, opts, message) => {
    if (message.$typeName !== typeName && foreignField !== void 0) {
      throw new FieldError(foreignField, `cannot use ${foreignField} with message ${message.$typeName}`, "ForeignFieldError");
    }
    for (let i = 0; i < fieldWriters.length; i++) {
      fieldWriters[i](writer, opts, message);
    }
    const unknown = message.$unknown;
    if (unknown !== void 0 && opts.writeUnknownFields) {
      for (let i = 0; i < unknown.length; i++) {
        const { no, wireType, data } = unknown[i];
        writer.tag(no, wireType).raw(data);
      }
    }
  };
  compiledWriters.set(desc, compiled);
  for (const field of sortedFields) {
    fieldWriters.push(compileField(field));
  }
  return compiled;
}
function compileField(field) {
  switch (field.fieldKind) {
    case "message":
    case "scalar":
    case "enum":
      return compileSingularField(field);
    case "list":
      return compileListField(field);
    case "map":
      return compileMapField(field);
  }
}
function compileSingularField(field) {
  const writeValue = compileSingularValue(field);
  const localName = field.localName;
  if (field.oneof) {
    const oneofLocalName = field.oneof.localName;
    return (writer, opts, message) => {
      const oneof = message[oneofLocalName];
      if (oneof.case === localName) {
        writeValue(writer, opts, oneof.value);
      }
    };
  }
  if (field.presence != IMPLICIT4) {
    const requiredError = field.presence == LEGACY_REQUIRED2 ? `cannot encode ${field} to binary: required field not set` : void 0;
    return (writer, opts, message) => {
      const value = message[localName];
      if (value !== void 0 && Object.prototype.hasOwnProperty.call(message, localName)) {
        writeValue(writer, opts, value);
      } else if (requiredError !== void 0) {
        throw new Error(requiredError);
      }
    };
  }
  if (field.fieldKind == "enum") {
    const zero = field.enum.values[0].number;
    return (writer, opts, message) => {
      const value = message[localName];
      if (value !== zero) {
        writeValue(writer, opts, value);
      }
    };
  }
  switch (field.scalar) {
    case ScalarType.BOOL:
      return (writer, opts, message) => {
        const value = message[localName];
        if (value !== false) {
          writeValue(writer, opts, value);
        }
      };
    case ScalarType.STRING:
      return (writer, opts, message) => {
        const value = message[localName];
        if (value !== "") {
          writeValue(writer, opts, value);
        }
      };
    case ScalarType.BYTES:
      return (writer, opts, message) => {
        const value = message[localName];
        if (!(value instanceof Uint8Array) || value.byteLength > 0) {
          writeValue(writer, opts, value);
        }
      };
    case ScalarType.DOUBLE:
    case ScalarType.FLOAT:
      return (writer, opts, message) => {
        const value = message[localName];
        if (!Object.is(value, 0)) {
          writeValue(writer, opts, value);
        }
      };
    default:
      return (writer, opts, message) => {
        const value = message[localName];
        if (value != 0) {
          writeValue(writer, opts, value);
        }
      };
  }
}
function compileSingularValue(field) {
  switch (field.fieldKind) {
    case "message": {
      const { toMessage } = localMessageMapper(field);
      const writeChild = compileChildWriter(field);
      return (writer, opts, value) => {
        writeChild(writer, opts, toMessage(value));
      };
    }
    case "scalar":
    case "enum": {
      const scalarType = field.fieldKind == "enum" ? ScalarType.INT32 : field.scalar;
      const fieldNo = field.number;
      const wireType = writeTypeOfScalar(scalarType);
      const writeScalar = compileScalarValue(scalarType, field.parent.typeName, field.name);
      return (writer, opts, value) => {
        writer.tag(fieldNo, wireType);
        writeScalar(writer, value);
      };
    }
  }
}
function compileListField(field) {
  const localName = field.localName;
  const fieldNo = field.number;
  switch (field.listKind) {
    case "message": {
      const { toMessage } = localMessageMapper(field);
      const writeChild = compileChildWriter(field);
      return (writer, opts, message) => {
        const items = message[localName];
        for (let i = 0; i < items.length; i++) {
          writeChild(writer, opts, toMessage(items[i]));
        }
      };
    }
    case "scalar":
    case "enum": {
      const scalarType = field.listKind == "enum" ? ScalarType.INT32 : field.scalar;
      const writeScalar = compileScalarValue(scalarType, field.parent.typeName, field.name);
      if (field.packed) {
        return (writer, opts, message) => {
          const items = message[localName];
          if (items.length == 0) {
            return;
          }
          writer.tag(fieldNo, WireType.LengthDelimited).fork();
          for (let i = 0; i < items.length; i++) {
            writeScalar(writer, items[i]);
          }
          writer.join();
        };
      }
      const wireType = writeTypeOfScalar(scalarType);
      return (writer, opts, message) => {
        const items = message[localName];
        for (let i = 0; i < items.length; i++) {
          writer.tag(fieldNo, wireType);
          writeScalar(writer, items[i]);
        }
      };
    }
  }
}
function compileMapField(field) {
  const localName = field.localName;
  const fieldNo = field.number;
  const writeKey = compileMapKey(field);
  if (field.mapKind == "message") {
    const { toMessage } = localMessageMapper(field);
    const writeMessage = compiledWriter(field.message);
    return (writer, opts, message) => {
      const record = message[localName];
      const keys = Object.keys(record);
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        writer.tag(fieldNo, WireType.LengthDelimited).fork();
        writeKey(writer, key);
        writer.tag(2, WireType.LengthDelimited).fork();
        writeMessage(writer, opts, toMessage(record[key]));
        writer.join();
        writer.join();
      }
    };
  }
  const scalarType = field.mapKind == "enum" ? ScalarType.INT32 : field.scalar;
  const valueWireType = writeTypeOfScalar(scalarType);
  const writeScalar = compileScalarValue(scalarType, field.parent.typeName, field.name);
  return (writer, opts, message) => {
    const record = message[localName];
    const keys = Object.keys(record);
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      writer.tag(fieldNo, WireType.LengthDelimited).fork();
      writeKey(writer, key);
      writer.tag(2, valueWireType);
      writeScalar(writer, record[key]);
      writer.join();
    }
  };
}
function compileMapKey(field) {
  const wireType = writeTypeOfScalar(field.mapKey);
  const writeScalar = compileScalarValue(field.mapKey, field.parent.typeName, field.name);
  const convertKey = compileMapKeyConverter(field.mapKey);
  return (writer, key) => {
    writer.tag(1, wireType);
    writeScalar(writer, convertKey(key));
  };
}
function compileMapKeyConverter(type) {
  switch (type) {
    case ScalarType.STRING:
      return (key) => key;
    case ScalarType.BOOL:
      return (key) => key === "true" ? true : key === "false" ? false : key;
    case ScalarType.UINT64:
    case ScalarType.FIXED64:
      return (key) => {
        try {
          return protoInt64.uParse(key);
        } catch (_a) {
          return key;
        }
      };
    case ScalarType.INT64:
    case ScalarType.SFIXED64:
    case ScalarType.SINT64:
      return (key) => {
        try {
          return protoInt64.parse(key);
        } catch (_a) {
          return key;
        }
      };
    default:
      return (key) => {
        const n = Number.parseInt(key);
        return Number.isFinite(n) ? n : key;
      };
  }
}
function compileScalarValue(type, messageName, fieldName) {
  const writeScalar = compileScalarWrite(type);
  return (writer, value) => {
    try {
      writeScalar(writer, value);
    } catch (e) {
      if (e instanceof Error) {
        throw new Error(`cannot encode field ${messageName}.${fieldName} to binary: ${e.message}`);
      }
      throw e;
    }
  };
}
function compileScalarWrite(type) {
  switch (type) {
    case ScalarType.STRING:
      return (writer, value) => writer.string(value);
    case ScalarType.BOOL:
      return (writer, value) => writer.bool(value);
    case ScalarType.DOUBLE:
      return (writer, value) => writer.double(value);
    case ScalarType.FLOAT:
      return (writer, value) => writer.float(value);
    case ScalarType.INT32:
      return (writer, value) => writer.int32(value);
    case ScalarType.INT64:
      return (writer, value) => writer.int64(value);
    case ScalarType.UINT64:
      return (writer, value) => writer.uint64(value);
    case ScalarType.FIXED64:
      return (writer, value) => writer.fixed64(value);
    case ScalarType.BYTES:
      return (writer, value) => writer.bytes(value);
    case ScalarType.FIXED32:
      return (writer, value) => writer.fixed32(value);
    case ScalarType.SFIXED32:
      return (writer, value) => writer.sfixed32(value);
    case ScalarType.SFIXED64:
      return (writer, value) => writer.sfixed64(value);
    case ScalarType.SINT64:
      return (writer, value) => writer.sint64(value);
    case ScalarType.UINT32:
      return (writer, value) => writer.uint32(value);
    case ScalarType.SINT32:
      return (writer, value) => writer.sint32(value);
  }
}
function writeField(writer, opts, msg, field) {
  compileField(field)(writer, opts, msg[unsafeLocal]);
}
function compileChildWriter(field) {
  const fieldNo = field.number;
  const writeMessage = compiledWriter(field.message);
  if (field.delimitedEncoding) {
    return (writer, opts, child) => {
      writer.tag(fieldNo, WireType.StartGroup);
      writeMessage(writer, opts, child);
      writer.tag(fieldNo, WireType.EndGroup);
    };
  }
  return (writer, opts, child) => {
    writer.tag(fieldNo, WireType.LengthDelimited).fork();
    writeMessage(writer, opts, child);
    writer.join();
  };
}
function writeTypeOfScalar(type) {
  switch (type) {
    case ScalarType.BYTES:
    case ScalarType.STRING:
      return WireType.LengthDelimited;
    case ScalarType.DOUBLE:
    case ScalarType.FIXED64:
    case ScalarType.SFIXED64:
      return WireType.Bit64;
    case ScalarType.FIXED32:
    case ScalarType.SFIXED32:
    case ScalarType.FLOAT:
      return WireType.Bit32;
    default:
      return WireType.Varint;
  }
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/reflect/reflect-check.js
function checkField(field, value) {
  const check = field.fieldKind == "list" ? isReflectList(value, field) : field.fieldKind == "map" ? isReflectMap(value, field) : checkSingular(field, value);
  if (check === true) {
    return void 0;
  }
  let reason;
  switch (field.fieldKind) {
    case "list":
      reason = `expected ${formatReflectList(field)}, got ${formatVal(value)}`;
      break;
    case "map":
      reason = `expected ${formatReflectMap(field)}, got ${formatVal(value)}`;
      break;
    default: {
      reason = reasonSingular(field, value, check);
    }
  }
  return new FieldError(field, reason);
}
function checkListItem(field, index, value) {
  const check = checkSingular(field, value);
  if (check !== true) {
    return new FieldError(field, `list item #${index + 1}: ${reasonSingular(field, value, check)}`);
  }
  return void 0;
}
function checkMapEntry(field, key, value) {
  const checkKey = checkScalarValue(field.mapKey)(key);
  if (checkKey !== true) {
    return new FieldError(field, `invalid map key: ${reasonSingular({ scalar: field.mapKey }, key, checkKey)}`);
  }
  const checkVal = checkSingular(field, value);
  if (checkVal !== true) {
    return new FieldError(field, `map entry ${formatVal(key)}: ${reasonSingular(field, value, checkVal)}`);
  }
  return void 0;
}
function checkSingular(field, value) {
  if (field.scalar !== void 0) {
    return checkScalarValue(field.scalar)(value);
  }
  if (field.enum !== void 0) {
    if (field.enum.open) {
      return checkScalarValue(ScalarType.INT32)(value);
    }
    return field.enum.values.some((v) => v.number === value);
  }
  return isReflectMessage(value, field.message);
}
function checkScalarValue(scalar) {
  switch (scalar) {
    case ScalarType.DOUBLE:
      return (value) => typeof value == "number";
    case ScalarType.FLOAT:
      return (value) => {
        if (typeof value != "number") {
          return false;
        }
        if (!Number.isFinite(Math.fround(value)) && Number.isFinite(value)) {
          return `${value.toFixed()} out of range`;
        }
        return true;
      };
    case ScalarType.INT32:
    case ScalarType.SFIXED32:
    case ScalarType.SINT32:
      return (value) => {
        if (typeof value !== "number" || !Number.isInteger(value)) {
          return false;
        }
        if (value > INT32_MAX || value < INT32_MIN) {
          return `${value.toFixed()} out of range`;
        }
        return true;
      };
    case ScalarType.FIXED32:
    case ScalarType.UINT32:
      return (value) => {
        if (typeof value !== "number" || !Number.isInteger(value)) {
          return false;
        }
        if (value > UINT32_MAX || value < 0) {
          return `${value.toFixed()} out of range`;
        }
        return true;
      };
    case ScalarType.BOOL:
      return (value) => typeof value == "boolean";
    case ScalarType.STRING:
      return (value) => {
        if (typeof value != "string") {
          return false;
        }
        return getTextEncoding().checkUtf8(value) || "invalid UTF8";
      };
    case ScalarType.BYTES:
      return (value) => value instanceof Uint8Array;
    case ScalarType.INT64:
    case ScalarType.SFIXED64:
    case ScalarType.SINT64:
      return (value) => {
        if (typeof value == "bigint" || typeof value == "number" || typeof value == "string" && value.length > 0) {
          try {
            protoInt64.parse(value);
            return true;
          } catch (_) {
            return `${value} out of range`;
          }
        }
        return false;
      };
    case ScalarType.FIXED64:
    case ScalarType.UINT64:
      return (value) => {
        if (typeof value == "bigint" || typeof value == "number" || typeof value == "string" && value.length > 0) {
          try {
            protoInt64.uParse(value);
            return true;
          } catch (_) {
            return `${value} out of range`;
          }
        }
        return false;
      };
  }
}
function reasonSingular(field, val, details) {
  details = typeof details == "string" ? `: ${details}` : `, got ${formatVal(val)}`;
  if (field.scalar !== void 0) {
    return `expected ${scalarTypeDescription(field.scalar)}` + details;
  }
  if (field.enum !== void 0) {
    return `expected ${field.enum.toString()}` + details;
  }
  return `expected ${formatReflectMessage(field.message)}` + details;
}
function formatVal(val) {
  switch (typeof val) {
    case "object":
      if (val === null) {
        return "null";
      }
      if (val instanceof Uint8Array) {
        return `Uint8Array(${val.length})`;
      }
      if (Array.isArray(val)) {
        return `Array(${val.length})`;
      }
      if (isReflectList(val)) {
        return formatReflectList(val.field());
      }
      if (isReflectMap(val)) {
        return formatReflectMap(val.field());
      }
      if (isReflectMessage(val)) {
        return formatReflectMessage(val.desc);
      }
      if (isMessage(val)) {
        return `message ${val.$typeName}`;
      }
      return "object";
    case "string":
      return val.length > 30 ? "string" : `"${val.split('"').join('\\"')}"`;
    case "boolean":
      return String(val);
    case "number":
      return String(val);
    case "bigint":
      return String(val) + "n";
    default:
      return typeof val;
  }
}
function formatReflectMessage(desc) {
  return `ReflectMessage (${desc.typeName})`;
}
function formatReflectList(field) {
  switch (field.listKind) {
    case "message":
      return `ReflectList (${field.message.toString()})`;
    case "enum":
      return `ReflectList (${field.enum.toString()})`;
    case "scalar":
      return `ReflectList (${ScalarType[field.scalar]})`;
  }
}
function formatReflectMap(field) {
  switch (field.mapKind) {
    case "message":
      return `ReflectMap (${ScalarType[field.mapKey]}, ${field.message.toString()})`;
    case "enum":
      return `ReflectMap (${ScalarType[field.mapKey]}, ${field.enum.toString()})`;
    case "scalar":
      return `ReflectMap (${ScalarType[field.mapKey]}, ${ScalarType[field.scalar]})`;
  }
}
function scalarTypeDescription(scalar) {
  switch (scalar) {
    case ScalarType.STRING:
      return "string";
    case ScalarType.BOOL:
      return "boolean";
    case ScalarType.INT64:
    case ScalarType.SINT64:
    case ScalarType.SFIXED64:
      return "bigint (int64)";
    case ScalarType.UINT64:
    case ScalarType.FIXED64:
      return "bigint (uint64)";
    case ScalarType.BYTES:
      return "Uint8Array";
    case ScalarType.DOUBLE:
      return "number (float64)";
    case ScalarType.FLOAT:
      return "number (float32)";
    case ScalarType.FIXED32:
    case ScalarType.UINT32:
      return "number (uint32)";
    case ScalarType.INT32:
    case ScalarType.SFIXED32:
    case ScalarType.SINT32:
      return "number (int32)";
  }
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/reflect/reflect.js
function reflect(messageDesc2, message, check = true) {
  return new ReflectMessageImpl(messageDesc2, message, check);
}
var messageSortedFields = /* @__PURE__ */ new WeakMap();
var ReflectMessageImpl = class {
  get sortedFields() {
    const cached = messageSortedFields.get(this.desc);
    if (cached) {
      return cached;
    }
    const sortedFields = this.desc.fields.concat().sort((a, b) => a.number - b.number);
    messageSortedFields.set(this.desc, sortedFields);
    return sortedFields;
  }
  constructor(messageDesc2, message, check = true) {
    this.lists = /* @__PURE__ */ new Map();
    this.maps = /* @__PURE__ */ new Map();
    this.check = check;
    this.desc = messageDesc2;
    this.message = this[unsafeLocal] = message !== null && message !== void 0 ? message : create(messageDesc2);
    this.fields = messageDesc2.fields;
    this.oneofs = messageDesc2.oneofs;
    this.members = messageDesc2.members;
  }
  findNumber(number) {
    if (!this._fieldsByNumber) {
      this._fieldsByNumber = new Map(this.desc.fields.map((f) => [f.number, f]));
    }
    return this._fieldsByNumber.get(number);
  }
  oneofCase(oneof) {
    assertOwn(this.message, oneof);
    return unsafeOneofCase(this.message, oneof);
  }
  isSet(field) {
    assertOwn(this.message, field);
    return unsafeIsSet(this.message, field);
  }
  clear(field) {
    assertOwn(this.message, field);
    unsafeClear(this.message, field);
  }
  get(field) {
    assertOwn(this.message, field);
    const value = unsafeGet(this.message, field);
    switch (field.fieldKind) {
      case "list":
        let list3 = this.lists.get(field);
        if (!list3 || list3[unsafeLocal] !== value) {
          this.lists.set(
            field,
            // biome-ignore lint/suspicious/noAssignInExpressions: no
            list3 = new ReflectListImpl(field, value, this.check)
          );
        }
        return list3;
      case "map":
        let map = this.maps.get(field);
        if (!map || map[unsafeLocal] !== value) {
          this.maps.set(
            field,
            // biome-ignore lint/suspicious/noAssignInExpressions: no
            map = new ReflectMapImpl(field, value, this.check)
          );
        }
        return map;
      case "message":
        return messageToReflect(field, value, this.check);
      case "scalar":
        return value === void 0 ? scalarZeroValue(field.scalar, false) : longToReflect(field, value);
      case "enum":
        return value !== null && value !== void 0 ? value : field.enum.values[0].number;
    }
  }
  set(field, value) {
    assertOwn(this.message, field);
    if (this.check) {
      const err = checkField(field, value);
      if (err) {
        throw err;
      }
    }
    let local;
    if (field.fieldKind == "message") {
      local = messageToLocal(field, value);
    } else if (isReflectMap(value) || isReflectList(value)) {
      local = value[unsafeLocal];
    } else {
      local = longToLocal(field, value);
    }
    unsafeSet(this.message, field, local);
  }
  getUnknown() {
    return this.message.$unknown;
  }
  setUnknown(value) {
    this.message.$unknown = value;
  }
};
function assertOwn(owner, member) {
  if (member.parent.typeName !== owner.$typeName) {
    throw new FieldError(member, `cannot use ${member.toString()} with message ${owner.$typeName}`, "ForeignFieldError");
  }
}
var ReflectListImpl = class {
  field() {
    return this._field;
  }
  get size() {
    return this._arr.length;
  }
  constructor(field, unsafeInput, check) {
    this._field = field;
    this._arr = this[unsafeLocal] = unsafeInput;
    this.check = check;
  }
  get(index) {
    const item = this._arr[index];
    return item === void 0 ? void 0 : listItemToReflect(this._field, item, this.check);
  }
  set(index, item) {
    if (index < 0 || index >= this._arr.length) {
      throw new FieldError(this._field, `list item #${index + 1}: out of range`);
    }
    if (this.check) {
      const err = checkListItem(this._field, index, item);
      if (err) {
        throw err;
      }
    }
    this._arr[index] = listItemToLocal(this._field, item);
  }
  add(item) {
    if (this.check) {
      const err = checkListItem(this._field, this._arr.length, item);
      if (err) {
        throw err;
      }
    }
    this._arr.push(listItemToLocal(this._field, item));
    return void 0;
  }
  clear() {
    this._arr.splice(0, this._arr.length);
  }
  [Symbol.iterator]() {
    return this.values();
  }
  keys() {
    return this._arr.keys();
  }
  *values() {
    for (const item of this._arr) {
      yield listItemToReflect(this._field, item, this.check);
    }
  }
  *entries() {
    for (let i = 0; i < this._arr.length; i++) {
      yield [i, listItemToReflect(this._field, this._arr[i], this.check)];
    }
  }
};
var ReflectMapImpl = class {
  constructor(field, unsafeInput, check = true) {
    this.obj = this[unsafeLocal] = unsafeInput !== null && unsafeInput !== void 0 ? unsafeInput : {};
    this.check = check;
    this._field = field;
  }
  field() {
    return this._field;
  }
  set(key, value) {
    if (this.check) {
      const err = checkMapEntry(this._field, key, value);
      if (err) {
        throw err;
      }
    }
    setOwn(this.obj, mapKeyToLocal(key), mapValueToLocal(this._field, value));
    return this;
  }
  delete(key) {
    const k = mapKeyToLocal(key);
    const has = Object.prototype.hasOwnProperty.call(this.obj, k);
    if (has) {
      delete this.obj[k];
    }
    return has;
  }
  clear() {
    for (const key of Object.keys(this.obj)) {
      delete this.obj[key];
    }
  }
  get(key) {
    let val = getOwn(this.obj, mapKeyToLocal(key));
    if (val !== void 0) {
      val = mapValueToReflect(this._field, val, this.check);
    }
    return val;
  }
  has(key) {
    return Object.prototype.hasOwnProperty.call(this.obj, mapKeyToLocal(key));
  }
  *keys() {
    for (const objKey of Object.keys(this.obj)) {
      yield mapKeyToReflect(objKey, this._field.mapKey);
    }
  }
  *entries() {
    for (const objEntry of Object.entries(this.obj)) {
      yield [
        mapKeyToReflect(objEntry[0], this._field.mapKey),
        mapValueToReflect(this._field, objEntry[1], this.check)
      ];
    }
  }
  [Symbol.iterator]() {
    return this.entries();
  }
  get size() {
    return Object.keys(this.obj).length;
  }
  *values() {
    for (const val of Object.values(this.obj)) {
      yield mapValueToReflect(this._field, val, this.check);
    }
  }
  forEach(callbackfn, thisArg) {
    for (const mapEntry of this.entries()) {
      callbackfn.call(thisArg, mapEntry[1], mapEntry[0], this);
    }
  }
};
function messageToLocal(field, value) {
  if (!isReflectMessage(value)) {
    return value;
  }
  if (isWrapper(value.message) && !field.oneof && field.fieldKind == "message") {
    return value.message.value;
  }
  if (value.desc.typeName == "google.protobuf.Struct" && field.parent.typeName != "google.protobuf.Value") {
    return wktStructToLocal(value.message);
  }
  return value.message;
}
function messageToReflect(field, value, check) {
  if (value !== void 0) {
    if (isWrapperDesc(field.message) && !field.oneof && field.fieldKind == "message") {
      value = {
        $typeName: field.message.typeName,
        value: longToReflect(field.message.fields[0], value)
      };
    } else if (field.message.typeName == "google.protobuf.Struct" && field.parent.typeName != "google.protobuf.Value" && isObject(value)) {
      value = wktStructToReflect(value);
    }
  }
  return new ReflectMessageImpl(field.message, value, check);
}
function listItemToLocal(field, value) {
  if (field.listKind == "message") {
    return messageToLocal(field, value);
  }
  return longToLocal(field, value);
}
function listItemToReflect(field, value, check) {
  if (field.listKind == "message") {
    return messageToReflect(field, value, check);
  }
  return longToReflect(field, value);
}
function mapValueToLocal(field, value) {
  if (field.mapKind == "message") {
    return messageToLocal(field, value);
  }
  return longToLocal(field, value);
}
function mapValueToReflect(field, value, check) {
  if (field.mapKind == "message") {
    return messageToReflect(field, value, check);
  }
  return value;
}
function mapKeyToLocal(key) {
  return typeof key == "string" || typeof key == "number" ? key : String(key);
}
function mapKeyToReflect(key, type) {
  switch (type) {
    case ScalarType.STRING:
      return key;
    case ScalarType.INT32:
    case ScalarType.FIXED32:
    case ScalarType.UINT32:
    case ScalarType.SFIXED32:
    case ScalarType.SINT32: {
      const n = Number.parseInt(key);
      if (Number.isFinite(n)) {
        return n;
      }
      break;
    }
    case ScalarType.BOOL:
      switch (key) {
        case "true":
          return true;
        case "false":
          return false;
      }
      break;
    case ScalarType.UINT64:
    case ScalarType.FIXED64:
      try {
        return protoInt64.uParse(key);
      } catch (_a) {
      }
      break;
    default:
      try {
        return protoInt64.parse(key);
      } catch (_b) {
      }
      break;
  }
  return key;
}
function longToReflect(field, value) {
  switch (field.scalar) {
    case ScalarType.INT64:
    case ScalarType.SFIXED64:
    case ScalarType.SINT64:
      if ("longAsString" in field && field.longAsString && typeof value == "string") {
        value = protoInt64.parse(value);
      }
      break;
    case ScalarType.FIXED64:
    case ScalarType.UINT64:
      if ("longAsString" in field && field.longAsString && typeof value == "string") {
        value = protoInt64.uParse(value);
      }
      break;
  }
  return value;
}
function longToLocal(field, value) {
  switch (field.scalar) {
    case ScalarType.INT64:
    case ScalarType.SFIXED64:
    case ScalarType.SINT64:
      if ("longAsString" in field && field.longAsString) {
        value = String(value);
      } else if (typeof value == "string" || typeof value == "number") {
        value = protoInt64.parse(value);
      }
      break;
    case ScalarType.FIXED64:
    case ScalarType.UINT64:
      if ("longAsString" in field && field.longAsString) {
        value = String(value);
      } else if (typeof value == "string" || typeof value == "number") {
        value = protoInt64.uParse(value);
      }
      break;
  }
  return value;
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/codegenv2/message.js
function messageDesc(file, path, ...paths) {
  return paths.reduce((acc, cur) => acc.nestedMessages[cur], file.messages[path]);
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/wkt/google/protobuf/descriptor_pb.js
var file_google_protobuf_descriptor = /* @__PURE__ */ boot({ "name": "google/protobuf/descriptor.proto", "package": "google.protobuf", "messageType": [{ "name": "FileDescriptorSet", "field": [{ "name": "file", "number": 1, "type": 11, "label": 3, "typeName": ".google.protobuf.FileDescriptorProto" }], "extensionRange": [{ "start": 536e6, "end": 536000001 }] }, { "name": "FileDescriptorProto", "field": [{ "name": "name", "number": 1, "type": 9, "label": 1 }, { "name": "package", "number": 2, "type": 9, "label": 1 }, { "name": "dependency", "number": 3, "type": 9, "label": 3 }, { "name": "public_dependency", "number": 10, "type": 5, "label": 3 }, { "name": "weak_dependency", "number": 11, "type": 5, "label": 3 }, { "name": "option_dependency", "number": 15, "type": 9, "label": 3 }, { "name": "message_type", "number": 4, "type": 11, "label": 3, "typeName": ".google.protobuf.DescriptorProto" }, { "name": "enum_type", "number": 5, "type": 11, "label": 3, "typeName": ".google.protobuf.EnumDescriptorProto" }, { "name": "service", "number": 6, "type": 11, "label": 3, "typeName": ".google.protobuf.ServiceDescriptorProto" }, { "name": "extension", "number": 7, "type": 11, "label": 3, "typeName": ".google.protobuf.FieldDescriptorProto" }, { "name": "options", "number": 8, "type": 11, "label": 1, "typeName": ".google.protobuf.FileOptions" }, { "name": "source_code_info", "number": 9, "type": 11, "label": 1, "typeName": ".google.protobuf.SourceCodeInfo" }, { "name": "syntax", "number": 12, "type": 9, "label": 1 }, { "name": "edition", "number": 14, "type": 14, "label": 1, "typeName": ".google.protobuf.Edition" }] }, { "name": "DescriptorProto", "field": [{ "name": "name", "number": 1, "type": 9, "label": 1 }, { "name": "field", "number": 2, "type": 11, "label": 3, "typeName": ".google.protobuf.FieldDescriptorProto" }, { "name": "extension", "number": 6, "type": 11, "label": 3, "typeName": ".google.protobuf.FieldDescriptorProto" }, { "name": "nested_type", "number": 3, "type": 11, "label": 3, "typeName": ".google.protobuf.DescriptorProto" }, { "name": "enum_type", "number": 4, "type": 11, "label": 3, "typeName": ".google.protobuf.EnumDescriptorProto" }, { "name": "extension_range", "number": 5, "type": 11, "label": 3, "typeName": ".google.protobuf.DescriptorProto.ExtensionRange" }, { "name": "oneof_decl", "number": 8, "type": 11, "label": 3, "typeName": ".google.protobuf.OneofDescriptorProto" }, { "name": "options", "number": 7, "type": 11, "label": 1, "typeName": ".google.protobuf.MessageOptions" }, { "name": "reserved_range", "number": 9, "type": 11, "label": 3, "typeName": ".google.protobuf.DescriptorProto.ReservedRange" }, { "name": "reserved_name", "number": 10, "type": 9, "label": 3 }, { "name": "visibility", "number": 11, "type": 14, "label": 1, "typeName": ".google.protobuf.SymbolVisibility" }], "nestedType": [{ "name": "ExtensionRange", "field": [{ "name": "start", "number": 1, "type": 5, "label": 1 }, { "name": "end", "number": 2, "type": 5, "label": 1 }, { "name": "options", "number": 3, "type": 11, "label": 1, "typeName": ".google.protobuf.ExtensionRangeOptions" }] }, { "name": "ReservedRange", "field": [{ "name": "start", "number": 1, "type": 5, "label": 1 }, { "name": "end", "number": 2, "type": 5, "label": 1 }] }] }, { "name": "ExtensionRangeOptions", "field": [{ "name": "uninterpreted_option", "number": 999, "type": 11, "label": 3, "typeName": ".google.protobuf.UninterpretedOption" }, { "name": "declaration", "number": 2, "type": 11, "label": 3, "typeName": ".google.protobuf.ExtensionRangeOptions.Declaration", "options": { "retention": 2 } }, { "name": "features", "number": 50, "type": 11, "label": 1, "typeName": ".google.protobuf.FeatureSet" }, { "name": "verification", "number": 3, "type": 14, "label": 1, "typeName": ".google.protobuf.ExtensionRangeOptions.VerificationState", "defaultValue": "UNVERIFIED", "options": { "retention": 2 } }], "nestedType": [{ "name": "Declaration", "field": [{ "name": "number", "number": 1, "type": 5, "label": 1 }, { "name": "full_name", "number": 2, "type": 9, "label": 1 }, { "name": "type", "number": 3, "type": 9, "label": 1 }, { "name": "reserved", "number": 5, "type": 8, "label": 1 }, { "name": "repeated", "number": 6, "type": 8, "label": 1 }] }], "enumType": [{ "name": "VerificationState", "value": [{ "name": "DECLARATION", "number": 0 }, { "name": "UNVERIFIED", "number": 1 }] }], "extensionRange": [{ "start": 990, "end": 999 }, { "start": 1e3, "end": 536870912 }] }, { "name": "FieldDescriptorProto", "field": [{ "name": "name", "number": 1, "type": 9, "label": 1 }, { "name": "number", "number": 3, "type": 5, "label": 1 }, { "name": "label", "number": 4, "type": 14, "label": 1, "typeName": ".google.protobuf.FieldDescriptorProto.Label" }, { "name": "type", "number": 5, "type": 14, "label": 1, "typeName": ".google.protobuf.FieldDescriptorProto.Type" }, { "name": "type_name", "number": 6, "type": 9, "label": 1 }, { "name": "extendee", "number": 2, "type": 9, "label": 1 }, { "name": "default_value", "number": 7, "type": 9, "label": 1 }, { "name": "oneof_index", "number": 9, "type": 5, "label": 1 }, { "name": "json_name", "number": 10, "type": 9, "label": 1 }, { "name": "options", "number": 8, "type": 11, "label": 1, "typeName": ".google.protobuf.FieldOptions" }, { "name": "proto3_optional", "number": 17, "type": 8, "label": 1 }], "enumType": [{ "name": "Type", "value": [{ "name": "TYPE_DOUBLE", "number": 1 }, { "name": "TYPE_FLOAT", "number": 2 }, { "name": "TYPE_INT64", "number": 3 }, { "name": "TYPE_UINT64", "number": 4 }, { "name": "TYPE_INT32", "number": 5 }, { "name": "TYPE_FIXED64", "number": 6 }, { "name": "TYPE_FIXED32", "number": 7 }, { "name": "TYPE_BOOL", "number": 8 }, { "name": "TYPE_STRING", "number": 9 }, { "name": "TYPE_GROUP", "number": 10 }, { "name": "TYPE_MESSAGE", "number": 11 }, { "name": "TYPE_BYTES", "number": 12 }, { "name": "TYPE_UINT32", "number": 13 }, { "name": "TYPE_ENUM", "number": 14 }, { "name": "TYPE_SFIXED32", "number": 15 }, { "name": "TYPE_SFIXED64", "number": 16 }, { "name": "TYPE_SINT32", "number": 17 }, { "name": "TYPE_SINT64", "number": 18 }] }, { "name": "Label", "value": [{ "name": "LABEL_OPTIONAL", "number": 1 }, { "name": "LABEL_REPEATED", "number": 3 }, { "name": "LABEL_REQUIRED", "number": 2 }] }] }, { "name": "OneofDescriptorProto", "field": [{ "name": "name", "number": 1, "type": 9, "label": 1 }, { "name": "options", "number": 2, "type": 11, "label": 1, "typeName": ".google.protobuf.OneofOptions" }] }, { "name": "EnumDescriptorProto", "field": [{ "name": "name", "number": 1, "type": 9, "label": 1 }, { "name": "value", "number": 2, "type": 11, "label": 3, "typeName": ".google.protobuf.EnumValueDescriptorProto" }, { "name": "options", "number": 3, "type": 11, "label": 1, "typeName": ".google.protobuf.EnumOptions" }, { "name": "reserved_range", "number": 4, "type": 11, "label": 3, "typeName": ".google.protobuf.EnumDescriptorProto.EnumReservedRange" }, { "name": "reserved_name", "number": 5, "type": 9, "label": 3 }, { "name": "visibility", "number": 6, "type": 14, "label": 1, "typeName": ".google.protobuf.SymbolVisibility" }], "nestedType": [{ "name": "EnumReservedRange", "field": [{ "name": "start", "number": 1, "type": 5, "label": 1 }, { "name": "end", "number": 2, "type": 5, "label": 1 }] }] }, { "name": "EnumValueDescriptorProto", "field": [{ "name": "name", "number": 1, "type": 9, "label": 1 }, { "name": "number", "number": 2, "type": 5, "label": 1 }, { "name": "options", "number": 3, "type": 11, "label": 1, "typeName": ".google.protobuf.EnumValueOptions" }] }, { "name": "ServiceDescriptorProto", "field": [{ "name": "name", "number": 1, "type": 9, "label": 1 }, { "name": "method", "number": 2, "type": 11, "label": 3, "typeName": ".google.protobuf.MethodDescriptorProto" }, { "name": "options", "number": 3, "type": 11, "label": 1, "typeName": ".google.protobuf.ServiceOptions" }] }, { "name": "MethodDescriptorProto", "field": [{ "name": "name", "number": 1, "type": 9, "label": 1 }, { "name": "input_type", "number": 2, "type": 9, "label": 1 }, { "name": "output_type", "number": 3, "type": 9, "label": 1 }, { "name": "options", "number": 4, "type": 11, "label": 1, "typeName": ".google.protobuf.MethodOptions" }, { "name": "client_streaming", "number": 5, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "server_streaming", "number": 6, "type": 8, "label": 1, "defaultValue": "false" }] }, { "name": "FileOptions", "field": [{ "name": "java_package", "number": 1, "type": 9, "label": 1 }, { "name": "java_outer_classname", "number": 8, "type": 9, "label": 1 }, { "name": "java_multiple_files", "number": 10, "type": 8, "label": 1, "defaultValue": "false", "options": {} }, { "name": "java_generate_equals_and_hash", "number": 20, "type": 8, "label": 1, "options": { "deprecated": true } }, { "name": "java_string_check_utf8", "number": 27, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "optimize_for", "number": 9, "type": 14, "label": 1, "typeName": ".google.protobuf.FileOptions.OptimizeMode", "defaultValue": "SPEED" }, { "name": "go_package", "number": 11, "type": 9, "label": 1 }, { "name": "cc_generic_services", "number": 16, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "java_generic_services", "number": 17, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "py_generic_services", "number": 18, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "deprecated", "number": 23, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "cc_enable_arenas", "number": 31, "type": 8, "label": 1, "defaultValue": "true", "options": {} }, { "name": "objc_class_prefix", "number": 36, "type": 9, "label": 1 }, { "name": "csharp_namespace", "number": 37, "type": 9, "label": 1 }, { "name": "swift_prefix", "number": 39, "type": 9, "label": 1 }, { "name": "php_class_prefix", "number": 40, "type": 9, "label": 1 }, { "name": "php_namespace", "number": 41, "type": 9, "label": 1 }, { "name": "php_metadata_namespace", "number": 44, "type": 9, "label": 1 }, { "name": "ruby_package", "number": 45, "type": 9, "label": 1 }, { "name": "features", "number": 50, "type": 11, "label": 1, "typeName": ".google.protobuf.FeatureSet" }, { "name": "uninterpreted_option", "number": 999, "type": 11, "label": 3, "typeName": ".google.protobuf.UninterpretedOption" }], "enumType": [{ "name": "OptimizeMode", "value": [{ "name": "SPEED", "number": 1 }, { "name": "CODE_SIZE", "number": 2 }, { "name": "LITE_RUNTIME", "number": 3 }] }], "extensionRange": [{ "start": 990, "end": 999 }, { "start": 1e3, "end": 536870912 }] }, { "name": "MessageOptions", "field": [{ "name": "message_set_wire_format", "number": 1, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "no_standard_descriptor_accessor", "number": 2, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "deprecated", "number": 3, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "map_entry", "number": 7, "type": 8, "label": 1 }, { "name": "deprecated_legacy_json_field_conflicts", "number": 11, "type": 8, "label": 1, "options": { "deprecated": true } }, { "name": "features", "number": 12, "type": 11, "label": 1, "typeName": ".google.protobuf.FeatureSet" }, { "name": "uninterpreted_option", "number": 999, "type": 11, "label": 3, "typeName": ".google.protobuf.UninterpretedOption" }], "extensionRange": [{ "start": 990, "end": 999 }, { "start": 1e3, "end": 536870912 }] }, { "name": "FieldOptions", "field": [{ "name": "ctype", "number": 1, "type": 14, "label": 1, "typeName": ".google.protobuf.FieldOptions.CType", "defaultValue": "STRING" }, { "name": "packed", "number": 2, "type": 8, "label": 1 }, { "name": "jstype", "number": 6, "type": 14, "label": 1, "typeName": ".google.protobuf.FieldOptions.JSType", "defaultValue": "JS_NORMAL" }, { "name": "lazy", "number": 5, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "unverified_lazy", "number": 15, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "deprecated", "number": 3, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "weak", "number": 10, "type": 8, "label": 1, "defaultValue": "false", "options": { "deprecated": true } }, { "name": "debug_redact", "number": 16, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "retention", "number": 17, "type": 14, "label": 1, "typeName": ".google.protobuf.FieldOptions.OptionRetention" }, { "name": "targets", "number": 19, "type": 14, "label": 3, "typeName": ".google.protobuf.FieldOptions.OptionTargetType" }, { "name": "edition_defaults", "number": 20, "type": 11, "label": 3, "typeName": ".google.protobuf.FieldOptions.EditionDefault" }, { "name": "features", "number": 21, "type": 11, "label": 1, "typeName": ".google.protobuf.FeatureSet" }, { "name": "feature_support", "number": 22, "type": 11, "label": 1, "typeName": ".google.protobuf.FieldOptions.FeatureSupport" }, { "name": "uninterpreted_option", "number": 999, "type": 11, "label": 3, "typeName": ".google.protobuf.UninterpretedOption" }], "nestedType": [{ "name": "EditionDefault", "field": [{ "name": "edition", "number": 3, "type": 14, "label": 1, "typeName": ".google.protobuf.Edition" }, { "name": "value", "number": 2, "type": 9, "label": 1 }] }, { "name": "FeatureSupport", "field": [{ "name": "edition_introduced", "number": 1, "type": 14, "label": 1, "typeName": ".google.protobuf.Edition" }, { "name": "edition_deprecated", "number": 2, "type": 14, "label": 1, "typeName": ".google.protobuf.Edition" }, { "name": "deprecation_warning", "number": 3, "type": 9, "label": 1 }, { "name": "edition_removed", "number": 4, "type": 14, "label": 1, "typeName": ".google.protobuf.Edition" }, { "name": "removal_error", "number": 5, "type": 9, "label": 1 }] }], "enumType": [{ "name": "CType", "value": [{ "name": "STRING", "number": 0 }, { "name": "CORD", "number": 1 }, { "name": "STRING_PIECE", "number": 2 }] }, { "name": "JSType", "value": [{ "name": "JS_NORMAL", "number": 0 }, { "name": "JS_STRING", "number": 1 }, { "name": "JS_NUMBER", "number": 2 }] }, { "name": "OptionRetention", "value": [{ "name": "RETENTION_UNKNOWN", "number": 0 }, { "name": "RETENTION_RUNTIME", "number": 1 }, { "name": "RETENTION_SOURCE", "number": 2 }] }, { "name": "OptionTargetType", "value": [{ "name": "TARGET_TYPE_UNKNOWN", "number": 0 }, { "name": "TARGET_TYPE_FILE", "number": 1 }, { "name": "TARGET_TYPE_EXTENSION_RANGE", "number": 2 }, { "name": "TARGET_TYPE_MESSAGE", "number": 3 }, { "name": "TARGET_TYPE_FIELD", "number": 4 }, { "name": "TARGET_TYPE_ONEOF", "number": 5 }, { "name": "TARGET_TYPE_ENUM", "number": 6 }, { "name": "TARGET_TYPE_ENUM_ENTRY", "number": 7 }, { "name": "TARGET_TYPE_SERVICE", "number": 8 }, { "name": "TARGET_TYPE_METHOD", "number": 9 }] }], "extensionRange": [{ "start": 990, "end": 999 }, { "start": 1e3, "end": 536870912 }] }, { "name": "OneofOptions", "field": [{ "name": "features", "number": 1, "type": 11, "label": 1, "typeName": ".google.protobuf.FeatureSet" }, { "name": "uninterpreted_option", "number": 999, "type": 11, "label": 3, "typeName": ".google.protobuf.UninterpretedOption" }], "extensionRange": [{ "start": 990, "end": 999 }, { "start": 1e3, "end": 536870912 }] }, { "name": "EnumOptions", "field": [{ "name": "allow_alias", "number": 2, "type": 8, "label": 1 }, { "name": "deprecated", "number": 3, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "deprecated_legacy_json_field_conflicts", "number": 6, "type": 8, "label": 1, "options": { "deprecated": true } }, { "name": "features", "number": 7, "type": 11, "label": 1, "typeName": ".google.protobuf.FeatureSet" }, { "name": "uninterpreted_option", "number": 999, "type": 11, "label": 3, "typeName": ".google.protobuf.UninterpretedOption" }], "extensionRange": [{ "start": 990, "end": 999 }, { "start": 1e3, "end": 536870912 }] }, { "name": "EnumValueOptions", "field": [{ "name": "deprecated", "number": 1, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "features", "number": 2, "type": 11, "label": 1, "typeName": ".google.protobuf.FeatureSet" }, { "name": "debug_redact", "number": 3, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "feature_support", "number": 4, "type": 11, "label": 1, "typeName": ".google.protobuf.FieldOptions.FeatureSupport" }, { "name": "uninterpreted_option", "number": 999, "type": 11, "label": 3, "typeName": ".google.protobuf.UninterpretedOption" }], "extensionRange": [{ "start": 990, "end": 999 }, { "start": 1e3, "end": 536870912 }] }, { "name": "ServiceOptions", "field": [{ "name": "features", "number": 34, "type": 11, "label": 1, "typeName": ".google.protobuf.FeatureSet" }, { "name": "deprecated", "number": 33, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "uninterpreted_option", "number": 999, "type": 11, "label": 3, "typeName": ".google.protobuf.UninterpretedOption" }], "extensionRange": [{ "start": 990, "end": 999 }, { "start": 1e3, "end": 536870912 }] }, { "name": "MethodOptions", "field": [{ "name": "deprecated", "number": 33, "type": 8, "label": 1, "defaultValue": "false" }, { "name": "idempotency_level", "number": 34, "type": 14, "label": 1, "typeName": ".google.protobuf.MethodOptions.IdempotencyLevel", "defaultValue": "IDEMPOTENCY_UNKNOWN" }, { "name": "features", "number": 35, "type": 11, "label": 1, "typeName": ".google.protobuf.FeatureSet" }, { "name": "uninterpreted_option", "number": 999, "type": 11, "label": 3, "typeName": ".google.protobuf.UninterpretedOption" }], "enumType": [{ "name": "IdempotencyLevel", "value": [{ "name": "IDEMPOTENCY_UNKNOWN", "number": 0 }, { "name": "NO_SIDE_EFFECTS", "number": 1 }, { "name": "IDEMPOTENT", "number": 2 }] }], "extensionRange": [{ "start": 990, "end": 999 }, { "start": 1e3, "end": 536870912 }] }, { "name": "UninterpretedOption", "field": [{ "name": "name", "number": 2, "type": 11, "label": 3, "typeName": ".google.protobuf.UninterpretedOption.NamePart" }, { "name": "identifier_value", "number": 3, "type": 9, "label": 1 }, { "name": "positive_int_value", "number": 4, "type": 4, "label": 1 }, { "name": "negative_int_value", "number": 5, "type": 3, "label": 1 }, { "name": "double_value", "number": 6, "type": 1, "label": 1 }, { "name": "string_value", "number": 7, "type": 12, "label": 1 }, { "name": "aggregate_value", "number": 8, "type": 9, "label": 1 }], "nestedType": [{ "name": "NamePart", "field": [{ "name": "name_part", "number": 1, "type": 9, "label": 2 }, { "name": "is_extension", "number": 2, "type": 8, "label": 2 }] }] }, { "name": "FeatureSet", "field": [{ "name": "field_presence", "number": 1, "type": 14, "label": 1, "typeName": ".google.protobuf.FeatureSet.FieldPresence", "options": { "retention": 1, "targets": [4, 1], "editionDefaults": [{ "value": "EXPLICIT", "edition": 900 }, { "value": "IMPLICIT", "edition": 999 }, { "value": "EXPLICIT", "edition": 1e3 }] } }, { "name": "enum_type", "number": 2, "type": 14, "label": 1, "typeName": ".google.protobuf.FeatureSet.EnumType", "options": { "retention": 1, "targets": [6, 1], "editionDefaults": [{ "value": "CLOSED", "edition": 900 }, { "value": "OPEN", "edition": 999 }] } }, { "name": "repeated_field_encoding", "number": 3, "type": 14, "label": 1, "typeName": ".google.protobuf.FeatureSet.RepeatedFieldEncoding", "options": { "retention": 1, "targets": [4, 1], "editionDefaults": [{ "value": "EXPANDED", "edition": 900 }, { "value": "PACKED", "edition": 999 }] } }, { "name": "utf8_validation", "number": 4, "type": 14, "label": 1, "typeName": ".google.protobuf.FeatureSet.Utf8Validation", "options": { "retention": 1, "targets": [4, 1], "editionDefaults": [{ "value": "NONE", "edition": 900 }, { "value": "VERIFY", "edition": 999 }] } }, { "name": "message_encoding", "number": 5, "type": 14, "label": 1, "typeName": ".google.protobuf.FeatureSet.MessageEncoding", "options": { "retention": 1, "targets": [4, 1], "editionDefaults": [{ "value": "LENGTH_PREFIXED", "edition": 900 }] } }, { "name": "json_format", "number": 6, "type": 14, "label": 1, "typeName": ".google.protobuf.FeatureSet.JsonFormat", "options": { "retention": 1, "targets": [3, 6, 1], "editionDefaults": [{ "value": "LEGACY_BEST_EFFORT", "edition": 900 }, { "value": "ALLOW", "edition": 999 }] } }, { "name": "enforce_naming_style", "number": 7, "type": 14, "label": 1, "typeName": ".google.protobuf.FeatureSet.EnforceNamingStyle", "options": { "retention": 2, "targets": [1, 2, 3, 4, 5, 6, 7, 8, 9], "editionDefaults": [{ "value": "STYLE_LEGACY", "edition": 900 }, { "value": "STYLE2024", "edition": 1001 }, { "value": "STYLE2026", "edition": 1002 }] } }, { "name": "default_symbol_visibility", "number": 8, "type": 14, "label": 1, "typeName": ".google.protobuf.FeatureSet.VisibilityFeature.DefaultSymbolVisibility", "options": { "retention": 2, "targets": [1], "editionDefaults": [{ "value": "EXPORT_ALL", "edition": 900 }, { "value": "EXPORT_TOP_LEVEL", "edition": 1001 }, { "value": "STRICT", "edition": 1002 }] } }, { "name": "enforce_proto_limits", "number": 9, "type": 14, "label": 1, "typeName": ".google.protobuf.FeatureSet.ProtoLimitsFeature.EnforceProtoLimits", "options": { "retention": 2, "targets": [6, 3, 4, 5], "editionDefaults": [{ "value": "LEGACY_NO_EXPLICIT_LIMITS", "edition": 900 }, { "value": "PROTO_LIMITS2026", "edition": 1002 }] } }], "nestedType": [{ "name": "VisibilityFeature", "enumType": [{ "name": "DefaultSymbolVisibility", "value": [{ "name": "DEFAULT_SYMBOL_VISIBILITY_UNKNOWN", "number": 0 }, { "name": "EXPORT_ALL", "number": 1 }, { "name": "EXPORT_TOP_LEVEL", "number": 2 }, { "name": "LOCAL_ALL", "number": 3 }, { "name": "STRICT", "number": 4 }] }] }, { "name": "ProtoLimitsFeature", "enumType": [{ "name": "EnforceProtoLimits", "value": [{ "name": "PROTO_LIMITS_UNKNOWN", "number": 0 }, { "name": "LEGACY_NO_EXPLICIT_LIMITS", "number": 1 }, { "name": "PROTO_LIMITS2026", "number": 2 }] }] }], "enumType": [{ "name": "FieldPresence", "value": [{ "name": "FIELD_PRESENCE_UNKNOWN", "number": 0 }, { "name": "EXPLICIT", "number": 1 }, { "name": "IMPLICIT", "number": 2 }, { "name": "LEGACY_REQUIRED", "number": 3 }] }, { "name": "EnumType", "value": [{ "name": "ENUM_TYPE_UNKNOWN", "number": 0 }, { "name": "OPEN", "number": 1 }, { "name": "CLOSED", "number": 2 }] }, { "name": "RepeatedFieldEncoding", "value": [{ "name": "REPEATED_FIELD_ENCODING_UNKNOWN", "number": 0 }, { "name": "PACKED", "number": 1 }, { "name": "EXPANDED", "number": 2 }] }, { "name": "Utf8Validation", "value": [{ "name": "UTF8_VALIDATION_UNKNOWN", "number": 0 }, { "name": "VERIFY", "number": 2 }, { "name": "NONE", "number": 3 }] }, { "name": "MessageEncoding", "value": [{ "name": "MESSAGE_ENCODING_UNKNOWN", "number": 0 }, { "name": "LENGTH_PREFIXED", "number": 1 }, { "name": "DELIMITED", "number": 2 }] }, { "name": "JsonFormat", "value": [{ "name": "JSON_FORMAT_UNKNOWN", "number": 0 }, { "name": "ALLOW", "number": 1 }, { "name": "LEGACY_BEST_EFFORT", "number": 2 }] }, { "name": "EnforceNamingStyle", "value": [{ "name": "ENFORCE_NAMING_STYLE_UNKNOWN", "number": 0 }, { "name": "STYLE2024", "number": 1 }, { "name": "STYLE_LEGACY", "number": 2 }, { "name": "STYLE2026", "number": 3 }] }], "extensionRange": [{ "start": 1e3, "end": 9995 }, { "start": 9995, "end": 1e4 }, { "start": 1e4, "end": 10001 }] }, { "name": "FeatureSetDefaults", "field": [{ "name": "defaults", "number": 1, "type": 11, "label": 3, "typeName": ".google.protobuf.FeatureSetDefaults.FeatureSetEditionDefault" }, { "name": "minimum_edition", "number": 4, "type": 14, "label": 1, "typeName": ".google.protobuf.Edition" }, { "name": "maximum_edition", "number": 5, "type": 14, "label": 1, "typeName": ".google.protobuf.Edition" }], "nestedType": [{ "name": "FeatureSetEditionDefault", "field": [{ "name": "edition", "number": 3, "type": 14, "label": 1, "typeName": ".google.protobuf.Edition" }, { "name": "overridable_features", "number": 4, "type": 11, "label": 1, "typeName": ".google.protobuf.FeatureSet" }, { "name": "fixed_features", "number": 5, "type": 11, "label": 1, "typeName": ".google.protobuf.FeatureSet" }] }] }, { "name": "SourceCodeInfo", "field": [{ "name": "location", "number": 1, "type": 11, "label": 3, "typeName": ".google.protobuf.SourceCodeInfo.Location" }], "nestedType": [{ "name": "Location", "field": [{ "name": "path", "number": 1, "type": 5, "label": 3, "options": { "packed": true } }, { "name": "span", "number": 2, "type": 5, "label": 3, "options": { "packed": true } }, { "name": "leading_comments", "number": 3, "type": 9, "label": 1 }, { "name": "trailing_comments", "number": 4, "type": 9, "label": 1 }, { "name": "leading_detached_comments", "number": 6, "type": 9, "label": 3 }] }], "extensionRange": [{ "start": 536e6, "end": 536000001 }] }, { "name": "GeneratedCodeInfo", "field": [{ "name": "annotation", "number": 1, "type": 11, "label": 3, "typeName": ".google.protobuf.GeneratedCodeInfo.Annotation" }], "nestedType": [{ "name": "Annotation", "field": [{ "name": "path", "number": 1, "type": 5, "label": 3, "options": { "packed": true } }, { "name": "source_file", "number": 2, "type": 9, "label": 1 }, { "name": "begin", "number": 3, "type": 5, "label": 1 }, { "name": "end", "number": 4, "type": 5, "label": 1 }, { "name": "semantic", "number": 5, "type": 14, "label": 1, "typeName": ".google.protobuf.GeneratedCodeInfo.Annotation.Semantic" }], "enumType": [{ "name": "Semantic", "value": [{ "name": "NONE", "number": 0 }, { "name": "SET", "number": 1 }, { "name": "ALIAS", "number": 2 }] }] }] }], "enumType": [{ "name": "Edition", "value": [{ "name": "EDITION_UNKNOWN", "number": 0 }, { "name": "EDITION_LEGACY", "number": 900 }, { "name": "EDITION_PROTO2", "number": 998 }, { "name": "EDITION_PROTO3", "number": 999 }, { "name": "EDITION_2023", "number": 1e3 }, { "name": "EDITION_2024", "number": 1001 }, { "name": "EDITION_2026", "number": 1002 }, { "name": "EDITION_UNSTABLE", "number": 9999 }, { "name": "EDITION_1_TEST_ONLY", "number": 1 }, { "name": "EDITION_2_TEST_ONLY", "number": 2 }, { "name": "EDITION_99997_TEST_ONLY", "number": 99997 }, { "name": "EDITION_99998_TEST_ONLY", "number": 99998 }, { "name": "EDITION_99999_TEST_ONLY", "number": 99999 }, { "name": "EDITION_MAX", "number": 2147483647 }] }, { "name": "SymbolVisibility", "value": [{ "name": "VISIBILITY_UNSET", "number": 0 }, { "name": "VISIBILITY_LOCAL", "number": 1 }, { "name": "VISIBILITY_EXPORT", "number": 2 }] }] });
var FileDescriptorProtoSchema = /* @__PURE__ */ messageDesc(file_google_protobuf_descriptor, 1);
var ExtensionRangeOptions_VerificationState;
(function(ExtensionRangeOptions_VerificationState2) {
  ExtensionRangeOptions_VerificationState2[ExtensionRangeOptions_VerificationState2["DECLARATION"] = 0] = "DECLARATION";
  ExtensionRangeOptions_VerificationState2[ExtensionRangeOptions_VerificationState2["UNVERIFIED"] = 1] = "UNVERIFIED";
})(ExtensionRangeOptions_VerificationState || (ExtensionRangeOptions_VerificationState = {}));
var FieldDescriptorProto_Type;
(function(FieldDescriptorProto_Type2) {
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["DOUBLE"] = 1] = "DOUBLE";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["FLOAT"] = 2] = "FLOAT";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["INT64"] = 3] = "INT64";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["UINT64"] = 4] = "UINT64";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["INT32"] = 5] = "INT32";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["FIXED64"] = 6] = "FIXED64";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["FIXED32"] = 7] = "FIXED32";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["BOOL"] = 8] = "BOOL";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["STRING"] = 9] = "STRING";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["GROUP"] = 10] = "GROUP";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["MESSAGE"] = 11] = "MESSAGE";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["BYTES"] = 12] = "BYTES";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["UINT32"] = 13] = "UINT32";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["ENUM"] = 14] = "ENUM";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["SFIXED32"] = 15] = "SFIXED32";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["SFIXED64"] = 16] = "SFIXED64";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["SINT32"] = 17] = "SINT32";
  FieldDescriptorProto_Type2[FieldDescriptorProto_Type2["SINT64"] = 18] = "SINT64";
})(FieldDescriptorProto_Type || (FieldDescriptorProto_Type = {}));
var FieldDescriptorProto_Label;
(function(FieldDescriptorProto_Label2) {
  FieldDescriptorProto_Label2[FieldDescriptorProto_Label2["OPTIONAL"] = 1] = "OPTIONAL";
  FieldDescriptorProto_Label2[FieldDescriptorProto_Label2["REPEATED"] = 3] = "REPEATED";
  FieldDescriptorProto_Label2[FieldDescriptorProto_Label2["REQUIRED"] = 2] = "REQUIRED";
})(FieldDescriptorProto_Label || (FieldDescriptorProto_Label = {}));
var FileOptions_OptimizeMode;
(function(FileOptions_OptimizeMode2) {
  FileOptions_OptimizeMode2[FileOptions_OptimizeMode2["SPEED"] = 1] = "SPEED";
  FileOptions_OptimizeMode2[FileOptions_OptimizeMode2["CODE_SIZE"] = 2] = "CODE_SIZE";
  FileOptions_OptimizeMode2[FileOptions_OptimizeMode2["LITE_RUNTIME"] = 3] = "LITE_RUNTIME";
})(FileOptions_OptimizeMode || (FileOptions_OptimizeMode = {}));
var FieldOptions_CType;
(function(FieldOptions_CType2) {
  FieldOptions_CType2[FieldOptions_CType2["STRING"] = 0] = "STRING";
  FieldOptions_CType2[FieldOptions_CType2["CORD"] = 1] = "CORD";
  FieldOptions_CType2[FieldOptions_CType2["STRING_PIECE"] = 2] = "STRING_PIECE";
})(FieldOptions_CType || (FieldOptions_CType = {}));
var FieldOptions_JSType;
(function(FieldOptions_JSType2) {
  FieldOptions_JSType2[FieldOptions_JSType2["JS_NORMAL"] = 0] = "JS_NORMAL";
  FieldOptions_JSType2[FieldOptions_JSType2["JS_STRING"] = 1] = "JS_STRING";
  FieldOptions_JSType2[FieldOptions_JSType2["JS_NUMBER"] = 2] = "JS_NUMBER";
})(FieldOptions_JSType || (FieldOptions_JSType = {}));
var FieldOptions_OptionRetention;
(function(FieldOptions_OptionRetention2) {
  FieldOptions_OptionRetention2[FieldOptions_OptionRetention2["RETENTION_UNKNOWN"] = 0] = "RETENTION_UNKNOWN";
  FieldOptions_OptionRetention2[FieldOptions_OptionRetention2["RETENTION_RUNTIME"] = 1] = "RETENTION_RUNTIME";
  FieldOptions_OptionRetention2[FieldOptions_OptionRetention2["RETENTION_SOURCE"] = 2] = "RETENTION_SOURCE";
})(FieldOptions_OptionRetention || (FieldOptions_OptionRetention = {}));
var FieldOptions_OptionTargetType;
(function(FieldOptions_OptionTargetType2) {
  FieldOptions_OptionTargetType2[FieldOptions_OptionTargetType2["TARGET_TYPE_UNKNOWN"] = 0] = "TARGET_TYPE_UNKNOWN";
  FieldOptions_OptionTargetType2[FieldOptions_OptionTargetType2["TARGET_TYPE_FILE"] = 1] = "TARGET_TYPE_FILE";
  FieldOptions_OptionTargetType2[FieldOptions_OptionTargetType2["TARGET_TYPE_EXTENSION_RANGE"] = 2] = "TARGET_TYPE_EXTENSION_RANGE";
  FieldOptions_OptionTargetType2[FieldOptions_OptionTargetType2["TARGET_TYPE_MESSAGE"] = 3] = "TARGET_TYPE_MESSAGE";
  FieldOptions_OptionTargetType2[FieldOptions_OptionTargetType2["TARGET_TYPE_FIELD"] = 4] = "TARGET_TYPE_FIELD";
  FieldOptions_OptionTargetType2[FieldOptions_OptionTargetType2["TARGET_TYPE_ONEOF"] = 5] = "TARGET_TYPE_ONEOF";
  FieldOptions_OptionTargetType2[FieldOptions_OptionTargetType2["TARGET_TYPE_ENUM"] = 6] = "TARGET_TYPE_ENUM";
  FieldOptions_OptionTargetType2[FieldOptions_OptionTargetType2["TARGET_TYPE_ENUM_ENTRY"] = 7] = "TARGET_TYPE_ENUM_ENTRY";
  FieldOptions_OptionTargetType2[FieldOptions_OptionTargetType2["TARGET_TYPE_SERVICE"] = 8] = "TARGET_TYPE_SERVICE";
  FieldOptions_OptionTargetType2[FieldOptions_OptionTargetType2["TARGET_TYPE_METHOD"] = 9] = "TARGET_TYPE_METHOD";
})(FieldOptions_OptionTargetType || (FieldOptions_OptionTargetType = {}));
var MethodOptions_IdempotencyLevel;
(function(MethodOptions_IdempotencyLevel2) {
  MethodOptions_IdempotencyLevel2[MethodOptions_IdempotencyLevel2["IDEMPOTENCY_UNKNOWN"] = 0] = "IDEMPOTENCY_UNKNOWN";
  MethodOptions_IdempotencyLevel2[MethodOptions_IdempotencyLevel2["NO_SIDE_EFFECTS"] = 1] = "NO_SIDE_EFFECTS";
  MethodOptions_IdempotencyLevel2[MethodOptions_IdempotencyLevel2["IDEMPOTENT"] = 2] = "IDEMPOTENT";
})(MethodOptions_IdempotencyLevel || (MethodOptions_IdempotencyLevel = {}));
var FeatureSet_VisibilityFeature_DefaultSymbolVisibility;
(function(FeatureSet_VisibilityFeature_DefaultSymbolVisibility2) {
  FeatureSet_VisibilityFeature_DefaultSymbolVisibility2[FeatureSet_VisibilityFeature_DefaultSymbolVisibility2["DEFAULT_SYMBOL_VISIBILITY_UNKNOWN"] = 0] = "DEFAULT_SYMBOL_VISIBILITY_UNKNOWN";
  FeatureSet_VisibilityFeature_DefaultSymbolVisibility2[FeatureSet_VisibilityFeature_DefaultSymbolVisibility2["EXPORT_ALL"] = 1] = "EXPORT_ALL";
  FeatureSet_VisibilityFeature_DefaultSymbolVisibility2[FeatureSet_VisibilityFeature_DefaultSymbolVisibility2["EXPORT_TOP_LEVEL"] = 2] = "EXPORT_TOP_LEVEL";
  FeatureSet_VisibilityFeature_DefaultSymbolVisibility2[FeatureSet_VisibilityFeature_DefaultSymbolVisibility2["LOCAL_ALL"] = 3] = "LOCAL_ALL";
  FeatureSet_VisibilityFeature_DefaultSymbolVisibility2[FeatureSet_VisibilityFeature_DefaultSymbolVisibility2["STRICT"] = 4] = "STRICT";
})(FeatureSet_VisibilityFeature_DefaultSymbolVisibility || (FeatureSet_VisibilityFeature_DefaultSymbolVisibility = {}));
var FeatureSet_ProtoLimitsFeature_EnforceProtoLimits;
(function(FeatureSet_ProtoLimitsFeature_EnforceProtoLimits2) {
  FeatureSet_ProtoLimitsFeature_EnforceProtoLimits2[FeatureSet_ProtoLimitsFeature_EnforceProtoLimits2["PROTO_LIMITS_UNKNOWN"] = 0] = "PROTO_LIMITS_UNKNOWN";
  FeatureSet_ProtoLimitsFeature_EnforceProtoLimits2[FeatureSet_ProtoLimitsFeature_EnforceProtoLimits2["LEGACY_NO_EXPLICIT_LIMITS"] = 1] = "LEGACY_NO_EXPLICIT_LIMITS";
  FeatureSet_ProtoLimitsFeature_EnforceProtoLimits2[FeatureSet_ProtoLimitsFeature_EnforceProtoLimits2["PROTO_LIMITS2026"] = 2] = "PROTO_LIMITS2026";
})(FeatureSet_ProtoLimitsFeature_EnforceProtoLimits || (FeatureSet_ProtoLimitsFeature_EnforceProtoLimits = {}));
var FeatureSet_FieldPresence;
(function(FeatureSet_FieldPresence2) {
  FeatureSet_FieldPresence2[FeatureSet_FieldPresence2["FIELD_PRESENCE_UNKNOWN"] = 0] = "FIELD_PRESENCE_UNKNOWN";
  FeatureSet_FieldPresence2[FeatureSet_FieldPresence2["EXPLICIT"] = 1] = "EXPLICIT";
  FeatureSet_FieldPresence2[FeatureSet_FieldPresence2["IMPLICIT"] = 2] = "IMPLICIT";
  FeatureSet_FieldPresence2[FeatureSet_FieldPresence2["LEGACY_REQUIRED"] = 3] = "LEGACY_REQUIRED";
})(FeatureSet_FieldPresence || (FeatureSet_FieldPresence = {}));
var FeatureSet_EnumType;
(function(FeatureSet_EnumType2) {
  FeatureSet_EnumType2[FeatureSet_EnumType2["ENUM_TYPE_UNKNOWN"] = 0] = "ENUM_TYPE_UNKNOWN";
  FeatureSet_EnumType2[FeatureSet_EnumType2["OPEN"] = 1] = "OPEN";
  FeatureSet_EnumType2[FeatureSet_EnumType2["CLOSED"] = 2] = "CLOSED";
})(FeatureSet_EnumType || (FeatureSet_EnumType = {}));
var FeatureSet_RepeatedFieldEncoding;
(function(FeatureSet_RepeatedFieldEncoding2) {
  FeatureSet_RepeatedFieldEncoding2[FeatureSet_RepeatedFieldEncoding2["REPEATED_FIELD_ENCODING_UNKNOWN"] = 0] = "REPEATED_FIELD_ENCODING_UNKNOWN";
  FeatureSet_RepeatedFieldEncoding2[FeatureSet_RepeatedFieldEncoding2["PACKED"] = 1] = "PACKED";
  FeatureSet_RepeatedFieldEncoding2[FeatureSet_RepeatedFieldEncoding2["EXPANDED"] = 2] = "EXPANDED";
})(FeatureSet_RepeatedFieldEncoding || (FeatureSet_RepeatedFieldEncoding = {}));
var FeatureSet_Utf8Validation;
(function(FeatureSet_Utf8Validation2) {
  FeatureSet_Utf8Validation2[FeatureSet_Utf8Validation2["UTF8_VALIDATION_UNKNOWN"] = 0] = "UTF8_VALIDATION_UNKNOWN";
  FeatureSet_Utf8Validation2[FeatureSet_Utf8Validation2["VERIFY"] = 2] = "VERIFY";
  FeatureSet_Utf8Validation2[FeatureSet_Utf8Validation2["NONE"] = 3] = "NONE";
})(FeatureSet_Utf8Validation || (FeatureSet_Utf8Validation = {}));
var FeatureSet_MessageEncoding;
(function(FeatureSet_MessageEncoding2) {
  FeatureSet_MessageEncoding2[FeatureSet_MessageEncoding2["MESSAGE_ENCODING_UNKNOWN"] = 0] = "MESSAGE_ENCODING_UNKNOWN";
  FeatureSet_MessageEncoding2[FeatureSet_MessageEncoding2["LENGTH_PREFIXED"] = 1] = "LENGTH_PREFIXED";
  FeatureSet_MessageEncoding2[FeatureSet_MessageEncoding2["DELIMITED"] = 2] = "DELIMITED";
})(FeatureSet_MessageEncoding || (FeatureSet_MessageEncoding = {}));
var FeatureSet_JsonFormat;
(function(FeatureSet_JsonFormat2) {
  FeatureSet_JsonFormat2[FeatureSet_JsonFormat2["JSON_FORMAT_UNKNOWN"] = 0] = "JSON_FORMAT_UNKNOWN";
  FeatureSet_JsonFormat2[FeatureSet_JsonFormat2["ALLOW"] = 1] = "ALLOW";
  FeatureSet_JsonFormat2[FeatureSet_JsonFormat2["LEGACY_BEST_EFFORT"] = 2] = "LEGACY_BEST_EFFORT";
})(FeatureSet_JsonFormat || (FeatureSet_JsonFormat = {}));
var FeatureSet_EnforceNamingStyle;
(function(FeatureSet_EnforceNamingStyle2) {
  FeatureSet_EnforceNamingStyle2[FeatureSet_EnforceNamingStyle2["ENFORCE_NAMING_STYLE_UNKNOWN"] = 0] = "ENFORCE_NAMING_STYLE_UNKNOWN";
  FeatureSet_EnforceNamingStyle2[FeatureSet_EnforceNamingStyle2["STYLE2024"] = 1] = "STYLE2024";
  FeatureSet_EnforceNamingStyle2[FeatureSet_EnforceNamingStyle2["STYLE_LEGACY"] = 2] = "STYLE_LEGACY";
  FeatureSet_EnforceNamingStyle2[FeatureSet_EnforceNamingStyle2["STYLE2026"] = 3] = "STYLE2026";
})(FeatureSet_EnforceNamingStyle || (FeatureSet_EnforceNamingStyle = {}));
var GeneratedCodeInfo_Annotation_Semantic;
(function(GeneratedCodeInfo_Annotation_Semantic2) {
  GeneratedCodeInfo_Annotation_Semantic2[GeneratedCodeInfo_Annotation_Semantic2["NONE"] = 0] = "NONE";
  GeneratedCodeInfo_Annotation_Semantic2[GeneratedCodeInfo_Annotation_Semantic2["SET"] = 1] = "SET";
  GeneratedCodeInfo_Annotation_Semantic2[GeneratedCodeInfo_Annotation_Semantic2["ALIAS"] = 2] = "ALIAS";
})(GeneratedCodeInfo_Annotation_Semantic || (GeneratedCodeInfo_Annotation_Semantic = {}));
var Edition;
(function(Edition2) {
  Edition2[Edition2["EDITION_UNKNOWN"] = 0] = "EDITION_UNKNOWN";
  Edition2[Edition2["EDITION_LEGACY"] = 900] = "EDITION_LEGACY";
  Edition2[Edition2["EDITION_PROTO2"] = 998] = "EDITION_PROTO2";
  Edition2[Edition2["EDITION_PROTO3"] = 999] = "EDITION_PROTO3";
  Edition2[Edition2["EDITION_2023"] = 1e3] = "EDITION_2023";
  Edition2[Edition2["EDITION_2024"] = 1001] = "EDITION_2024";
  Edition2[Edition2["EDITION_2026"] = 1002] = "EDITION_2026";
  Edition2[Edition2["EDITION_UNSTABLE"] = 9999] = "EDITION_UNSTABLE";
  Edition2[Edition2["EDITION_1_TEST_ONLY"] = 1] = "EDITION_1_TEST_ONLY";
  Edition2[Edition2["EDITION_2_TEST_ONLY"] = 2] = "EDITION_2_TEST_ONLY";
  Edition2[Edition2["EDITION_99997_TEST_ONLY"] = 99997] = "EDITION_99997_TEST_ONLY";
  Edition2[Edition2["EDITION_99998_TEST_ONLY"] = 99998] = "EDITION_99998_TEST_ONLY";
  Edition2[Edition2["EDITION_99999_TEST_ONLY"] = 99999] = "EDITION_99999_TEST_ONLY";
  Edition2[Edition2["EDITION_MAX"] = 2147483647] = "EDITION_MAX";
})(Edition || (Edition = {}));
var SymbolVisibility;
(function(SymbolVisibility2) {
  SymbolVisibility2[SymbolVisibility2["VISIBILITY_UNSET"] = 0] = "VISIBILITY_UNSET";
  SymbolVisibility2[SymbolVisibility2["VISIBILITY_LOCAL"] = 1] = "VISIBILITY_LOCAL";
  SymbolVisibility2[SymbolVisibility2["VISIBILITY_EXPORT"] = 2] = "VISIBILITY_EXPORT";
})(SymbolVisibility || (SymbolVisibility = {}));

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/from-binary.js
function makeReadContext(options2) {
  return Object.assign(Object.assign({ readUnknownFields: true, recursionLimit: 100 }, options2), { depth: 0 });
}
function fromBinary(schema, bytes2, options2) {
  const message = create(schema);
  compiledReader(schema).read(message, new BinaryReader(bytes2), makeReadContext(options2), bytes2.byteLength);
  return message;
}
var compiledReaders = /* @__PURE__ */ new WeakMap();
function compiledReader(desc) {
  let compiled = compiledReaders.get(desc);
  if (compiled === void 0) {
    compiled = compileMessage2(desc);
  }
  return compiled;
}
function compileMessage2(desc) {
  const descString = String(desc);
  const fieldReaders = /* @__PURE__ */ new Map();
  const compiled = {
    read: compileMessageReader(descString, fieldReaders),
    readGroup: compileGroupReader(descString, fieldReaders)
  };
  compiledReaders.set(desc, compiled);
  for (const field of desc.fields) {
    fieldReaders.set(field.number, compileFieldReader(field));
  }
  return compiled;
}
function compileMessageReader(descString, fieldReaders) {
  return (message, reader, ctx, length) => {
    var _a;
    if (++ctx.depth > ctx.recursionLimit) {
      throw new Error(`cannot decode ${descString} from binary: maximum recursion depth of ${ctx.recursionLimit} reached`);
    }
    const end = reader.pos + length;
    const unknownFields = (_a = message.$unknown) !== null && _a !== void 0 ? _a : [];
    while (reader.pos < end) {
      const [fieldNo, wireType] = reader.tag();
      const fieldReader = fieldReaders.get(fieldNo);
      if (fieldReader === void 0) {
        const data = reader.skip(wireType, fieldNo, ctx.recursionLimit - ctx.depth);
        if (ctx.readUnknownFields) {
          unknownFields.push({ no: fieldNo, wireType, data });
        }
        continue;
      }
      fieldReader(message, reader, ctx, wireType);
    }
    if (unknownFields.length > 0) {
      message.$unknown = unknownFields;
    }
    ctx.depth--;
  };
}
function compileGroupReader(descString, fieldReaders) {
  return (message, reader, ctx, fieldNo) => {
    var _a;
    if (++ctx.depth > ctx.recursionLimit) {
      throw new Error(`cannot decode ${descString} from binary: maximum recursion depth of ${ctx.recursionLimit} reached`);
    }
    let recordFieldNo;
    let wireType;
    const unknownFields = (_a = message.$unknown) !== null && _a !== void 0 ? _a : [];
    while (reader.pos < reader.len) {
      [recordFieldNo, wireType] = reader.tag();
      if (wireType == WireType.EndGroup) {
        break;
      }
      const fieldReader = fieldReaders.get(recordFieldNo);
      if (fieldReader === void 0) {
        const data = reader.skip(wireType, recordFieldNo, ctx.recursionLimit - ctx.depth);
        if (ctx.readUnknownFields) {
          unknownFields.push({ no: recordFieldNo, wireType, data });
        }
        continue;
      }
      fieldReader(message, reader, ctx, wireType);
    }
    if (wireType != WireType.EndGroup || recordFieldNo !== fieldNo) {
      throw new Error("invalid end group tag");
    }
    if (unknownFields.length > 0) {
      message.$unknown = unknownFields;
    }
    ctx.depth--;
  };
}
function readField(message, reader, field, wireType, ctx) {
  compileFieldReader(field)(message[unsafeLocal], reader, ctx, wireType);
}
function compileFieldReader(field) {
  switch (field.fieldKind) {
    case "scalar":
      return compileScalarFieldReader(field);
    case "enum":
      return compileEnumFieldReader(field);
    case "message":
      return compileMessageFieldReader(field);
    case "list":
      return compileListFieldReader(field);
    case "map":
      return compileMapFieldReader(field);
  }
}
function compileScalarFieldReader(field) {
  const readScalar = compileScalarReader(field.scalar, field.utf8Validation, field.longAsString);
  const localName = field.localName;
  if (field.oneof) {
    const oneofLocalName = field.oneof.localName;
    return (message, reader) => {
      message[oneofLocalName] = {
        case: localName,
        value: readScalar(reader)
      };
    };
  }
  return (message, reader) => {
    message[localName] = readScalar(reader);
  };
}
function compileEnumFieldReader(field) {
  var _a;
  const localName = field.localName;
  const oneofLocalName = (_a = field.oneof) === null || _a === void 0 ? void 0 : _a.localName;
  if (field.enum.open) {
    if (oneofLocalName !== void 0) {
      return (message, reader) => {
        message[oneofLocalName] = { case: localName, value: reader.int32() };
      };
    }
    return (message, reader) => {
      message[localName] = reader.int32();
    };
  }
  const values = field.enum.values;
  const fieldNo = field.number;
  return (message, reader, ctx, wireType) => {
    var _a2;
    const val = reader.int32();
    if (values.some((v) => v.number === val)) {
      if (oneofLocalName !== void 0) {
        message[oneofLocalName] = { case: localName, value: val };
      } else {
        message[localName] = val;
      }
    } else if (ctx.readUnknownFields) {
      const bytes2 = [];
      varint32write(val, bytes2);
      const unknownFields = (_a2 = message.$unknown) !== null && _a2 !== void 0 ? _a2 : [];
      unknownFields.push({
        no: fieldNo,
        wireType,
        data: new Uint8Array(bytes2)
      });
      message.$unknown = unknownFields;
    }
  };
}
function compileMessageFieldReader(field) {
  const localName = field.localName;
  const { toMessage, toLocal } = localMessageMapper(field);
  const readChild = compileChildReader(field);
  if (field.oneof) {
    const oneofLocalName = field.oneof.localName;
    return (message, reader, ctx) => {
      const oneof = message[oneofLocalName];
      const child = toMessage(oneof.case === localName ? oneof.value : void 0);
      readChild(child, reader, ctx);
      message[oneofLocalName] = { case: localName, value: toLocal(child) };
    };
  }
  return (message, reader, ctx) => {
    const child = toMessage(message[localName]);
    readChild(child, reader, ctx);
    message[localName] = toLocal(child);
  };
}
function compileChildReader(field) {
  const compiledChild = compiledReader(field.message);
  if (field.delimitedEncoding) {
    const fieldNo = field.number;
    return (child, reader, ctx) => compiledChild.readGroup(child, reader, ctx, fieldNo);
  }
  return (child, reader, ctx) => compiledChild.read(child, reader, ctx, reader.uint32());
}
function compileListFieldReader(field) {
  const localName = field.localName;
  if (field.listKind == "message") {
    const { toMessage, toLocal } = localMessageMapper(field);
    const readChild = compileChildReader(field);
    return (message, reader, ctx) => {
      const child = toMessage(void 0);
      readChild(child, reader, ctx);
      message[localName].push(toLocal(child));
    };
  }
  const scalarType = field.listKind == "enum" ? ScalarType.INT32 : field.scalar;
  const longAsString = field.listKind == "scalar" ? field.longAsString : false;
  const readScalar = compileScalarReader(scalarType, field.utf8Validation, longAsString);
  const packedPossible = scalarType != ScalarType.STRING && scalarType != ScalarType.BYTES;
  return (message, reader, ctx, wireType) => {
    const items = message[localName];
    if (wireType == WireType.LengthDelimited && packedPossible) {
      const end = reader.uint32() + reader.pos;
      while (reader.pos < end) {
        items.push(readScalar(reader));
      }
    } else {
      items.push(readScalar(reader));
    }
  };
}
function compileMapFieldReader(field) {
  const localName = field.localName;
  const readKey = compileScalarReader(field.mapKey, field.utf8Validation, false);
  const keyZero = scalarZeroValue(field.mapKey, false);
  let readValue;
  let valueDefault;
  switch (field.mapKind) {
    case "scalar": {
      const scalar = field.scalar;
      const readScalar = compileScalarReader(scalar, field.utf8Validation, false);
      readValue = (reader) => readScalar(reader);
      if (scalar == ScalarType.BYTES) {
        valueDefault = () => new Uint8Array(0);
      } else {
        const zero = scalarZeroValue(scalar, false);
        valueDefault = () => zero;
      }
      break;
    }
    case "enum": {
      const zero = field.enum.values[0].number;
      readValue = (reader) => reader.int32();
      valueDefault = () => zero;
      break;
    }
    case "message": {
      const { toMessage, toLocal } = localMessageMapper(field);
      const readChild = compiledReader(field.message).read;
      readValue = (reader, ctx, existing) => {
        const child = toMessage(existing);
        readChild(child, reader, ctx, reader.uint32());
        return toLocal(child);
      };
      valueDefault = () => toLocal(toMessage(void 0));
      break;
    }
  }
  return (message, reader, ctx) => {
    const record = message[localName];
    let key;
    let val;
    const len = reader.uint32();
    const end = reader.pos + len;
    while (reader.pos < end) {
      const [fieldNo] = reader.tag();
      switch (fieldNo) {
        case 1:
          key = readKey(reader);
          break;
        case 2:
          val = readValue(reader, ctx, val);
          break;
      }
    }
    if (key === void 0) {
      key = keyZero;
    }
    if (val === void 0) {
      val = valueDefault();
    }
    setOwn(record, key, val);
  };
}
function compileScalarReader(type, utf8Validation, longAsString) {
  switch (type) {
    case ScalarType.STRING:
      return (reader) => reader.string(utf8Validation);
    case ScalarType.BOOL:
      return (reader) => reader.bool();
    case ScalarType.DOUBLE:
      return (reader) => reader.double();
    case ScalarType.FLOAT:
      return (reader) => reader.float();
    case ScalarType.INT32:
      return (reader) => reader.int32();
    case ScalarType.INT64:
      if (longAsString) {
        return (reader) => String(reader.int64());
      }
      return (reader) => reader.int64();
    case ScalarType.UINT64:
      if (longAsString) {
        return (reader) => String(reader.uint64());
      }
      return (reader) => reader.uint64();
    case ScalarType.FIXED64:
      if (longAsString) {
        return (reader) => String(reader.fixed64());
      }
      return (reader) => reader.fixed64();
    case ScalarType.BYTES:
      return (reader) => reader.bytes();
    case ScalarType.FIXED32:
      return (reader) => reader.fixed32();
    case ScalarType.SFIXED32:
      return (reader) => reader.sfixed32();
    case ScalarType.SFIXED64:
      if (longAsString) {
        return (reader) => String(reader.sfixed64());
      }
      return (reader) => reader.sfixed64();
    case ScalarType.SINT64:
      if (longAsString) {
        return (reader) => String(reader.sint64());
      }
      return (reader) => reader.sint64();
    case ScalarType.UINT32:
      return (reader) => reader.uint32();
    case ScalarType.SINT32:
      return (reader) => reader.sint32();
  }
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/codegenv2/file.js
function fileDesc(b64, imports) {
  var _a;
  const root = fromBinary(FileDescriptorProtoSchema, base64Decode(b64));
  root.messageType.forEach(restoreJsonNames);
  root.dependency = (_a = imports === null || imports === void 0 ? void 0 : imports.map((f) => f.proto.name)) !== null && _a !== void 0 ? _a : [];
  const reg = createFileRegistry(root, (protoFileName) => imports === null || imports === void 0 ? void 0 : imports.find((f) => f.proto.name === protoFileName));
  return reg.getFile(root.name);
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/codegenv2/service.js
function serviceDesc(file, path, ...paths) {
  if (paths.length > 0) {
    throw new Error();
  }
  return file.services[path];
}

// demo/web/src/gen/turbo_typed_annotations_pb.ts
var file_turbo_typed_annotations = /* @__PURE__ */ fileDesc("Ch10dXJib190eXBlZF9hbm5vdGF0aW9ucy5wcm90bxIPdHVyYm8ucmVzdWx0LnYxIjgKCE1vZGVsUmVmEgoKAmlkGAEgASgJEhAKCHJldmlzaW9uGAIgASgJEg4KBnNoYTI1NhgDIAEoCSKIAQoIVmlld0luZm8SEAoIaW5zdGFuY2UYASABKAkSMAoGc3RhdHVzGAIgASgOMiAudHVyYm8ucmVzdWx0LnYxLkxheWVyV2lyZVN0YXR1cxIVCg1zdGF0dXNfZGV0YWlsGAMgASgJEhEKCWFubm90YXRvchgEIAEoCRIOCgZwYXJlbnQYBSABKAkieAobSW5zcGVjdERvY3VtZW50R3JhcGhSZXF1ZXN0Eg4KBmRvY19pZBgBIAEoCRINCgVncmFwaBgCIAEoDBIMCgR2aWV3GAMgASgNEiwKB29mZnNldHMYBCABKA4yGy50dXJiby5yZXN1bHQudjEuT2Zmc2V0TW9kZSJDCg1FeGVjdXRpb25JbmZvEg8KB2JhY2tlbmQYASABKAkSDgoGZGV2aWNlGAIgASgJEhEKCXByZWNpc2lvbhgDIAEoCSLeAQoPTW9kZWxQcm92ZW5hbmNlEhEKCW1vZGVsX3JlZhgBIAEoCRIRCglhbm5vdGF0b3IYAiABKAkSDwoHdmVyc2lvbhgDIAEoCRITCgtjb25maWdfaGFzaBgEIAEoBBIuCgZidW5kbGUYBSABKAsyGS50dXJiby5yZXN1bHQudjEuTW9kZWxSZWZIAIgBARI2CglleGVjdXRpb24YBiABKAsyHi50dXJiby5yZXN1bHQudjEuRXhlY3V0aW9uSW5mb0gBiAEBQgkKB19idW5kbGVCDAoKX2V4ZWN1dGlvbiIqCgxTcGFuRnJhZ21lbnQSDQoFc3RhcnQYASABKA0SCwoDZW5kGAIgASgNIkEKD0xheWVySW5kZXhCb3VuZBINCgVsYXllchgBIAEoCRIQCghpbnN0YW5jZRgCIAEoCRINCgVib3VuZBgDIAEoDSL8AQoTVHlwZWREb2N1bWVudFJlc3VsdBIOCgZkb2NfaWQYASABKAkSEAoIcmF3X3RleHQYAiABKAkSDAoEdmlldxgDIAEoDRIsCgdvZmZzZXRzGAQgASgOMhsudHVyYm8ucmVzdWx0LnYxLk9mZnNldE1vZGUSMAoGbW9kZWxzGAUgAygLMiAudHVyYm8ucmVzdWx0LnYxLk1vZGVsUHJvdmVuYW5jZRIoCgV2aWV3cxgGIAMoCzIZLnR1cmJvLnJlc3VsdC52MS5WaWV3SW5mbxIrCgZsYXllcnMYByADKAsyGy50dXJiby5yZXN1bHQudjEuVHlwZWRMYXllciLPAgoKVHlwZWRMYXllchIKCgJpZBgBIAEoCRIQCghpbnN0YW5jZRgCIAEoCRIwCgZzdGF0dXMYAyABKA4yIC50dXJiby5yZXN1bHQudjEuTGF5ZXJXaXJlU3RhdHVzEhUKDXN0YXR1c19kZXRhaWwYBCABKAkSKgoFc2NvcGUYBSABKA4yGy50dXJiby5yZXN1bHQudjEuQW5jaG9yS2luZBISCgptb2RlbF9yZWZzGAYgAygJEhkKDGFuY2hvcl9sYXllchgHIAEoCUgAiAEBEjUKC2Fubm90YXRpb25zGAggAygLMiAudHVyYm8ucmVzdWx0LnYxLlR5cGVkQW5ub3RhdGlvbhI3Cg1pbmRleGVzX2xheWVyGAkgAygLMiAudHVyYm8ucmVzdWx0LnYxLkxheWVySW5kZXhCb3VuZEIPCg1fYW5jaG9yX2xheWVyIoUCCgpFbnRpdHlWb3RlEhEKCWFubm90YXRvchgBIAEoCRIWCgltb2RlbF9yZWYYAiABKAlIAIgBARIsCgVzcGFucxgDIAMoCzIdLnR1cmJvLnJlc3VsdC52MS5TcGFuRnJhZ21lbnQSGAoLcHJvYmFiaWxpdHkYBCABKAFIAYgBARIjChZjYWxpYnJhdGVkX3Byb2JhYmlsaXR5GAUgASgBSAKIAQESFwoKYWJzdGVudGlvbhgGIAEoCUgDiAEBQgwKCl9tb2RlbF9yZWZCDgoMX3Byb2JhYmlsaXR5QhkKF19jYWxpYnJhdGVkX3Byb2JhYmlsaXR5Qg0KC19hYnN0ZW50aW9uIjEKElNlbnRlbmNlQW5ub3RhdGlvbhINCgVpbmRleBgBIAEoDRIMCgR0ZXh0GAIgASgJIlIKD1Rva2VuQW5ub3RhdGlvbhIbCg5zZW50ZW5jZV9pbmRleBgBIAEoDUgAiAEBEg8KB3N1cmZhY2UYAiABKAlCEQoPX3NlbnRlbmNlX2luZGV4IocBCg5Xb3JkQW5ub3RhdGlvbhITCgt0b2tlbl9zdGFydBgBIAEoDRIRCgl0b2tlbl9lbmQYAiABKA0SMwoHc3Vic3BhbhgDIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5TcGFuRnJhZ21lbnRIAIgBARIMCgRmb3JtGAQgASgJQgoKCF9zdWJzcGFuImAKEVN1YndvcmRBbm5vdGF0aW9uEhAKCHZvY2FiX2lkGAEgASgFEhgKC3Rva2VuX2luZGV4GAIgASgNSACIAQESDwoHc3VyZmFjZRgDIAEoCUIOCgxfdG9rZW5faW5kZXgiRQoQUG9zVGFnQW5ub3RhdGlvbhIOCgZ0YWdzZXQYASABKAkSCwoDdGFnGAIgASgJEhQKDHRhcmdldF9pbmRleBgDIAEoDSJBChdNb3JwaEZlYXR1cmVzQW5ub3RhdGlvbhIUCgx0YXJnZXRfaW5kZXgYASABKA0SEAoIZmVhdHVyZXMYAiABKAkiNgoPTGVtbWFBbm5vdGF0aW9uEhQKDHRhcmdldF9pbmRleBgBIAEoDRINCgVsZW1tYRgCIAEoCSIzCg5TdGVtQW5ub3RhdGlvbhITCgt0b2tlbl9pbmRleBgBIAEoDRIMCgRzdGVtGAIgASgJIoABCg5UZXJtQW5ub3RhdGlvbhITCgt0b2tlbl9pbmRleBgBIAEoDRIPCgdzdXJmYWNlGAIgASgJEg4KBmZvbGRlZBgDIAEoCRIRCgRzdGVtGAQgASgJSACIAQESEgoFbGVtbWEYBSABKAlIAYgBAUIHCgVfc3RlbUIICgZfbGVtbWEiRgoSQ2h1bmtUYWdBbm5vdGF0aW9uEg4KBnRhZ3NldBgBIAEoCRILCgN0YWcYAiABKAkSEwoLdG9rZW5faW5kZXgYAyABKA0iXQoVUGhyYXNlQ2h1bmtBbm5vdGF0aW9uEg4KBnRhZ3NldBgBIAEoCRIMCgRraW5kGAIgASgJEhMKC3Rva2VuX3N0YXJ0GAMgASgNEhEKCXRva2VuX2VuZBgEIAEoDSKYAQoQUGhyYXNlQW5ub3RhdGlvbhIOCgZ0YWdzZXQYASABKAkSDAoEa2luZBgCIAEoCRITCgt0b2tlbl9zdGFydBgDIAEoDRIRCgl0b2tlbl9lbmQYBCABKA0SEgoKaGVhZF90b2tlbhgFIAEoDRIZCgxwYXJlbnRfaW5kZXgYBiABKA1IAIgBAUIPCg1fcGFyZW50X2luZGV4InYKFERlcGVuZGVuY3lBbm5vdGF0aW9uEg4KBnRhZ3NldBgBIAEoCRIUCgx0YXJnZXRfaW5kZXgYAiABKA0SFwoKaGVhZF9pbmRleBgDIAEoDUgAiAEBEhAKCHJlbGF0aW9uGAQgASgJQg0KC19oZWFkX2luZGV4IrUBChBFbnRpdHlBbm5vdGF0aW9uEg4KBnRhZ3NldBgBIAEoCRIMCgRraW5kGAIgASgJEhMKC3Rva2VuX3N0YXJ0GAMgASgNEhEKCXRva2VuX2VuZBgEIAEoDRIRCgljYW5vbmljYWwYBSABKAkSEgoFa2JfaWQYBiABKAlIAIgBARIqCgV2b3RlcxgHIAMoCzIbLnR1cmJvLnJlc3VsdC52MS5FbnRpdHlWb3RlQggKBl9rYl9pZCJhChJSZWxhdGlvbkFubm90YXRpb24SDgoGdGFnc2V0GAEgASgJEgwKBGtpbmQYAiABKAkSFgoOc3ViamVjdF9lbnRpdHkYAyABKA0SFQoNb2JqZWN0X2VudGl0eRgEIAEoDSL3AQoWQ29yZWZNZW50aW9uQW5ub3RhdGlvbhIWCg5zZW50ZW5jZV9pbmRleBgBIAEoDRITCgt0b2tlbl9zdGFydBgCIAEoDRIRCgl0b2tlbl9lbmQYAyABKA0SEgoKaGVhZF90b2tlbhgEIAEoDRIZCgxlbnRpdHlfaW5kZXgYBSABKA1IAIgBARIMCgRraW5kGAYgASgJEg4KBm9yaWdpbhgHIAEoCRIOCgZudW1iZXIYCCABKAkSDgoGZ2VuZGVyGAkgASgJEg8KB2FuaW1hY3kYCiABKAkSDgoGcGVyc29uGAsgASgJQg8KDV9lbnRpdHlfaW5kZXgiRwoUQ29yZWZDaGFpbkFubm90YXRpb24SFgoOcmVwcmVzZW50YXRpdmUYASABKAkSFwoPbWVudGlvbl9pbmRpY2VzGAIgAygNIiQKEVNwZWFrZXJBbm5vdGF0aW9uEg8KB3NwZWFrZXIYASABKAki9wEKFldvcmROZXRTZW5zZUFubm90YXRpb24SEwoLdG9rZW5faW5kZXgYASABKA0SFQoNc3luc2V0X29mZnNldBgCIAEoDRILCgNwb3MYAyABKAkSDwoHbGV4bmFtZRgEIAEoCRINCgVnbG9zcxgFIAEoCRIOCgZsZW1tYXMYBiADKAkSGAoQaHlwZXJueW1fb2Zmc2V0cxgHIAMoDRIRCglzeW5zZXRfaWQYCCABKAkSEAoIc2Vuc2VfaWQYCSABKAkSDQoFbGVtbWEYCiABKAkSEgoKc2Vuc2VfcmFuaxgLIAEoDRISCgpsZXhpY29uX2lkGAwgASgJIvoBChpMZXhpY2FsRXhwYW5zaW9uQW5ub3RhdGlvbhITCgt0b2tlbl9pbmRleBgBIAEoDRIQCghzeW5vbnltcxgCIAMoCRIRCgloeXBlcm55bXMYAyADKAkSEwoLZGVyaXZhdGlvbnMYBCADKAkSEwoLbW9ycGhfcm9vdHMYBSADKAkSDAoEdGVybRgGIAEoCRIQCghyZWxhdGlvbhgHIAEoDRINCgVkZXB0aBgIIAEoDRISCgpzZW5zZV9yYW5rGAkgASgNEg4KBndlaWdodBgKIAEoARIRCglzeW5zZXRfaWQYCyABKAkSEgoKbGV4aWNvbl9pZBgMIAEoCSKEAQoWR2xvc3NhcnlUZXJtQW5ub3RhdGlvbhIUCgxjYW5vbmljYWxfaWQYASABKAkSFwoPcHJlZmVycmVkX2xhYmVsGAIgASgJEhMKC2VudHJ5X2luZGV4GAMgASgNEhMKC3Rva2VuX3N0YXJ0GAQgASgNEhEKCXRva2VuX2VuZBgFIAEoDSLIAQoQTnVtYmVyQW5ub3RhdGlvbhIzCgxudW1lcmFsX3NwYW4YASABKAsyHS50dXJiby5yZXN1bHQudjEuU3BhbkZyYWdtZW50EhAKCG5lZ2F0aXZlGAIgASgIEhMKC2NvZWZmaWNpZW50GAMgASgJEg0KBXNjYWxlGAQgASgNEhIKCm11bHRpcGxpZXIYBSABKAkSDgoGbWFya2VyGAYgASgJEhAKCG5vdGF0aW9uGAcgASgJEhMKC3BsYWluX3ZhbHVlGAggASgJIjEKEU9yZGluYWxBbm5vdGF0aW9uEgwKBGJhc2UYASABKAkSDgoGb2Zmc2V0GAIgASgDIm8KD01vbmV5QW5ub3RhdGlvbhIQCghjdXJyZW5jeRgBIAEoCRIQCghuZWdhdGl2ZRgCIAEoCBITCgtjb2VmZmljaWVudBgDIAEoCRINCgVzY2FsZRgEIAEoBRIUCgxwbGFpbl9hbW91bnQYBSABKAkigQEKElF1YW50aXR5QW5ub3RhdGlvbhIMCgR1bml0GAEgASgJEhAKCG5lZ2F0aXZlGAIgASgIEhMKC2NvZWZmaWNpZW50GAMgASgJEg0KBXNjYWxlGAQgASgFEhMKC3BsYWluX3ZhbHVlGAUgASgJEhIKCmlzX3BlcmNlbnQYBiABKAgicQoSVGVtcG9yYWxBbm5vdGF0aW9uEhMKC2dyYW51bGFyaXR5GAEgASgJEg4KBm9yaWdpbhgCIAEoCRILCgNpc28YAyABKAkSDAoEeWVhchgEIAEoBRIOCgZwZXJpb2QYBSABKA0SCwoDZGF5GAYgASgNIsMBCg5UaW1lQW5ub3RhdGlvbhIMCgRraW5kGAEgASgJEg0KBWdyYWluGAIgASgJEhAKCG1vZGlmaWVyGAMgASgJEhIKBXN0YXJ0GAQgASgDSACIAQESEAoDZW5kGAUgASgDSAGIAQESGAoLZHVyYXRpb25fbXMYBiABKANIAogBARIRCglhbWJpZ3VvdXMYByABKAgSDQoFdGltZXgYCCABKAlCCAoGX3N0YXJ0QgYKBF9lbmRCDgoMX2R1cmF0aW9uX21zIk8KFkRvY3VtZW50RGF0ZUFubm90YXRpb24SDAoEeWVhchgBIAEoBRINCgVtb250aBgCIAEoDRILCgNkYXkYAyABKA0SCwoDaXNvGAQgASgJIu4BCg9QbGFjZVJlc29sdXRpb24SDgoGc291cmNlGAEgASgJEhEKCXJlY29yZF9pZBgCIAEoCRIWCg5jYW5vbmljYWxfbmFtZRgDIAEoCRIVCg1mZWF0dXJlX2NsYXNzGAQgASgJEhQKDGNvdW50cnlfY29kZRgFIAEoCRITCgthZG1pbjFfY29kZRgGIAEoCRITCgthZG1pbjJfY29kZRgHIAEoCRIQCghsYXRpdHVkZRgIIAEoARIRCglsb25naXR1ZGUYCSABKAESEgoKcG9wdWxhdGlvbhgKIAEoBBIQCgh0aW1lem9uZRgLIAEoCSI8ChJQbGFjZUFuY2VzdG9yUHJvdG8SCgoCaWQYASABKAkSDAoEbmFtZRgCIAEoCRIMCgRraW5kGAMgASgJIkoKEENvbnRhaW5tZW50Q2hhaW4SNgoJYW5jZXN0b3JzGAEgAygLMiMudHVyYm8ucmVzdWx0LnYxLlBsYWNlQW5jZXN0b3JQcm90byIxCgpSZWdpb25Wb3RlEhQKDGNvdW50cnlfY29kZRgBIAEoCRINCgVzaGFyZRgCIAEoASIxChJMYW5ndWFnZUFubm90YXRpb24SDAoEY29kZRgBIAEoCRINCgVzY29yZRgCIAEoASJKChZMYW5ndWFnZVNwYW5Bbm5vdGF0aW9uEgwKBGNvZGUYASABKAkSDgoGc2NyaXB0GAIgASgJEhIKCmNvbmZpZGVuY2UYAyABKAEiMgoSQ2F0ZWdvcnlTY29yZVByb3RvEg0KBWxhYmVsGAEgASgJEg0KBXNjb3JlGAIgASgBIpIBChJDYXRlZ29yeUFubm90YXRpb24SDgoGdGFnc2V0GAEgASgJEhIKCmJlc3RfbGFiZWwYAiABKAkSEQoJYWJzdGFpbmVkGAMgASgIEhAKCHNlbGVjdGVkGAQgAygJEjMKBnNjb3JlcxgFIAMoCzIjLnR1cmJvLnJlc3VsdC52MS5DYXRlZ29yeVNjb3JlUHJvdG8iXAoTU2VudGltZW50QW5ub3RhdGlvbhIQCghwb2xhcml0eRgBIAEoCRIQCghwb3NpdGl2ZRgCIAEoARIQCghuZWdhdGl2ZRgDIAEoARIPCgduZXV0cmFsGAQgASgBIkwKE0xhYmVsVG9rZW5TcGFuUHJvdG8SDQoFbGFiZWwYASABKAkSEwoLdG9rZW5fc3RhcnQYAiABKA0SEQoJdG9rZW5fZW5kGAMgASgNIoYBChFFbW90aW9uQW5ub3RhdGlvbhITCgt0YXJnZXRfa2luZBgBIAEoCRIUCgx0YXJnZXRfaW5kZXgYAiABKA0SDgoGc2NvcmVzGAMgAygBEjYKCGV2aWRlbmNlGAQgAygLMiQudHVyYm8ucmVzdWx0LnYxLkxhYmVsVG9rZW5TcGFuUHJvdG8ilgIKEEFzcGVjdEFubm90YXRpb24SGAoLdG9rZW5fc3RhcnQYASABKA1IAIgBARIWCgl0b2tlbl9lbmQYAiABKA1IAYgBARIbCg5zZW50ZW5jZV9pbmRleBgDIAEoDUgCiAEBEhUKCGNhdGVnb3J5GAQgASgJSAOIAQESEAoIcG9sYXJpdHkYBSABKAkSEAoIcG9zaXRpdmUYBiABKAESEAoIbmVnYXRpdmUYByABKAESDwoHbmV1dHJhbBgIIAEoARIXCg9vcGluaW9uX2luZGljZXMYCSADKA1CDgoMX3Rva2VuX3N0YXJ0QgwKCl90b2tlbl9lbmRCEQoPX3NlbnRlbmNlX2luZGV4QgsKCV9jYXRlZ29yeSJxChFPcGluaW9uQW5ub3RhdGlvbhITCgt0b2tlbl9zdGFydBgBIAEoDRIRCgl0b2tlbl9lbmQYAiABKA0SEAoIcG9sYXJpdHkYAyABKAESEQoJaW50ZW5zaXR5GAQgASgBEg8KB25lZ2F0ZWQYBSABKAgi8wEKFEluY2l2aWxpdHlBbm5vdGF0aW9uEhMKC3RhcmdldF9raW5kGAEgASgJEhQKDHRhcmdldF9pbmRleBgCIAEoDRIOCgZpbnN1bHQYAyABKAESEQoJcHJvZmFuaXR5GAQgASgBEg4KBnRocmVhdBgFIAEoARIXCg9pZGVudGl0eV9hdHRhY2sYBiABKAESEwoLaW50b2xlcmFuY2UYByABKAESFwoPc2V2ZXJlX3RveGljaXR5GAggASgBEjYKCGV2aWRlbmNlGAkgAygLMiQudHVyYm8ucmVzdWx0LnYxLkxhYmVsVG9rZW5TcGFuUHJvdG8ipQIKFFByYWdtYXRpY3NBbm5vdGF0aW9uEhYKDnNlbnRlbmNlX2luZGV4GAEgASgNEg0KBXNjb3JlGAIgASgBEiEKFHBvc2l0aXZlX3Rva2VuX3N0YXJ0GAMgASgNSACIAQESHwoScG9zaXRpdmVfdG9rZW5fZW5kGAQgASgNSAGIAQESIQoUbmVnYXRpdmVfdG9rZW5fc3RhcnQYBSABKA1IAogBARIfChJuZWdhdGl2ZV90b2tlbl9lbmQYBiABKA1IA4gBAUIXChVfcG9zaXRpdmVfdG9rZW5fc3RhcnRCFQoTX3Bvc2l0aXZlX3Rva2VuX2VuZEIXChVfbmVnYXRpdmVfdG9rZW5fc3RhcnRCFQoTX25lZ2F0aXZlX3Rva2VuX2VuZCJHCgpQaWlNZW50aW9uEgwKBGtpbmQYASABKAkSFwoPY29uZmlkZW5jZV90aWVyGAIgASgJEhIKCm5vcm1hbGl6ZWQYAyABKAkiLAoJTm9pc2VTcGFuEhAKCHNldmVyaXR5GAEgASgJEg0KBXNjb3JlGAIgASgBIhwKDFRleHRBcnRpZmFjdBIMCgRraW5kGAEgASgJIuYBCg1FbWJlZGRlZEFzc2V0Eg4KBmZvcm1hdBgBIAEoCRISCgptZWRpYV90eXBlGAIgASgJEhUKDWRlY2xhcmVkX3R5cGUYAyABKAkSEAoIZW5jb2RpbmcYBCABKAkSMwoMcGF5bG9hZF9zcGFuGAUgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudBIVCg1kZWNvZGVkX2J5dGVzGAYgASgEEhIKBXdpZHRoGAcgASgNSACIAQESEwoGaGVpZ2h0GAggASgNSAGIAQFCCAoGX3dpZHRoQgkKB19oZWlnaHQiPwoSU3BlbGxpbmdDb3JyZWN0aW9uEhIKCnN1Z2dlc3Rpb24YASABKAkSFQoNZWRpdF9kaXN0YW5jZRgCIAEoDSIlChVQbGFjZWhvbGRlckFubm90YXRpb24SDAoEa2luZBgBIAEoCSI9Cg9FbW9qaUFubm90YXRpb24SDAoEa2luZBgBIAEoCRIMCgRuYW1lGAIgASgJEg4KBnJlZ2lvbhgDIAEoCSIoChdQcm90ZWN0ZWRTcGFuQW5ub3RhdGlvbhINCgVpbmRleBgBIAEoDSImChZBYmJyZXZpYXRpb25Bbm5vdGF0aW9uEgwKBGZvcm0YASABKAkiWQoPTWF0Y2hBbm5vdGF0aW9uEhMKC3Rva2VuX3N0YXJ0GAEgASgNEhEKCXRva2VuX2VuZBgCIAEoDRINCgVsYWJlbBgDIAEoCRIPCgdydWxlX2lkGAQgASgNIlgKFUJvaWxlcnBsYXRlQW5ub3RhdGlvbhIUCgx0YXJnZXRfaW5kZXgYASABKA0SEQoEa2luZBgCIAEoCUgAiAEBEg0KBXNjb3JlGAMgASgCQgcKBV9raW5kIkAKFENvbmZpZGVuY2VBbm5vdGF0aW9uEhQKDHRhcmdldF9pbmRleBgBIAEoDRISCgpjb25maWRlbmNlGAIgASgBIlQKFlN0b3B3b3JkRmxhZ0Fubm90YXRpb24SEwoLdG9rZW5faW5kZXgYASABKA0SEAoIbGFuZ3VhZ2UYAiABKAkSEwoLaXNfc3RvcHdvcmQYAyABKAgiNwoSV29yZFR5cGVBbm5vdGF0aW9uEhMKC3Rva2VuX2luZGV4GAEgASgNEgwKBGtpbmQYAiABKAkilgEKEVJlYWRpbmdBbm5vdGF0aW9uEhMKC3Rva2VuX2luZGV4GAEgASgNEhQKB3JlYWRpbmcYAiABKAlIAIgBARIaCg1wcm9udW5jaWF0aW9uGAMgASgJSAGIAQESEgoFbGVtbWEYBCABKAlIAogBAUIKCghfcmVhZGluZ0IQCg5fcHJvbnVuY2lhdGlvbkIICgZfbGVtbWEiVAoRS2V5d29yZEFubm90YXRpb24SDgoGcGhyYXNlGAEgASgJEg0KBXNjb3JlGAIgASgBEhEKCWFsZ29yaXRobRgDIAEoCRINCgV0ZXJtcxgEIAEoDSJSChhTb3VyY2VFbmNvZGluZ0Fubm90YXRpb24SEAoIZW5jb2RpbmcYASABKAkSEgoKY29uZmlkZW5jZRgCIAEoAhIQCghldmlkZW5jZRgDIAEoCSJUChBWZWN0b3JBbm5vdGF0aW9uEhQKDHRhcmdldF9pbmRleBgBIAEoDRILCgNkaW0YAiABKA0SDQoFZW1wdHkYAyABKAgSDgoGdmFsdWVzGAQgAygCIscBCg9DaHVua0Fubm90YXRpb24SDQoFbGV2ZWwYASABKA0SGQoMcGFyZW50X2luZGV4GAIgASgNSACIAQESFgoOc2VudGVuY2Vfc3RhcnQYAyABKA0SFAoMc2VudGVuY2VfZW5kGAQgASgNEhIKCmN1dF9yZWFzb24YBSABKAkSDAoEY29zdBgGIAEoAxIUCgxtb2RlbF90b2tlbnMYByABKA0SEwoLb3Zlcl9idWRnZXQYCCABKAhCDwoNX3BhcmVudF9pbmRleCJfCg9MZXZlbEFubm90YXRpb24SDAoEa2luZBgBIAEoCRISCgp1bml0X2luZGV4GAIgASgNEhkKDHBhcmVudF9pbmRleBgDIAEoDUgAiAEBQg8KDV9wYXJlbnRfaW5kZXgiSgoSQm91bmRhcnlBbm5vdGF0aW9uEhYKDnNlbnRlbmNlX2luZGV4GAEgASgNEgwKBGNvc3QYAiABKAUSDgoGcmVhc29uGAMgASgJIl4KE1N0cnVjdHVyZUFubm90YXRpb24SDAoEa2luZBgBIAEoCRIZCgxwYXJlbnRfaW5kZXgYAiABKA1IAIgBARINCgVkZXB0aBgDIAEoDUIPCg1fcGFyZW50X2luZGV4InIKEEFuc3dlckFubm90YXRpb24SDQoFc2NvcmUYASABKAESEwoLcHJvYmFiaWxpdHkYAiABKAESEgoKbnVsbF9zY29yZRgDIAEoARIYChBudWxsX3Byb2JhYmlsaXR5GAQgASgBEgwKBHRleHQYBSABKAkiUwoORWRnZUFubm90YXRpb24SEgoKZnJvbV9pbmRleBgBIAEoDRIQCgh0b19pbmRleBgCIAEoDRIMCgRraW5kGAMgASgJEg0KBXNjb3JlGAQgASgBIiIKD0dyb3VwQW5ub3RhdGlvbhIPCgdtZW1iZXJzGAEgAygNIvEBChZMaW5rZWRFbnRpdHlBbm5vdGF0aW9uEhQKDGVudGl0eV9pbmRleBgBIAEoDRISCgVrYl9pZBgCIAEoCUgAiAEBEhMKC3Byb3ZpZGVyX2lkGAMgASgJEhYKDmNhbm9uaWNhbF9uYW1lGAQgASgJEhUKDW1hdGNoZWRfYWxpYXMYBSABKAkSDQoFc2NvcmUYBiABKAESFwoKbmlsX3JlYXNvbhgHIAEoCUgBiAEBEhgKC25pbF9jbHVzdGVyGAggASgNSAKIAQFCCAoGX2tiX2lkQg0KC19uaWxfcmVhc29uQg4KDF9uaWxfY2x1c3RlciKNAQoUQ2FsaWJyYXRlZEFubm90YXRpb24SFAoMdGFyZ2V0X2luZGV4GAEgASgNEgsKA3JhdxgCIAMoARISCgpjYWxpYnJhdGVkGAMgAygBEhYKDnByZWRpY3Rpb25fc2V0GAQgAygNEhcKCmFic3RlbnRpb24YBSABKAlIAIgBAUINCgtfYWJzdGVudGlvbiIzChtUb2tlblZlY3RvcldpbmRvd0Fubm90YXRpb24SFAoMd2luZG93X2luZGV4GAEgASgNIlgKE1V0ZjE2U3BhbkFubm90YXRpb24SFAoMdGFyZ2V0X2luZGV4GAEgASgNEg8KB3N0YXJ0MTYYAiABKA0SDQoFZW5kMTYYAyABKA0SCwoDY3V0GAQgASgIIpcBChJNb2RhbGl0eUFubm90YXRpb24SEAoIcG9sYXJpdHkYASABKAkSEQoJZXBpc3RlbWljGAIgASgJEg8KB2Rlb250aWMYAyABKAkSLwoDY3VlGAQgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudEgAiAEBEhIKCmNvbmZpZGVuY2UYBSABKAJCBgoEX2N1ZSKqAQoYQ29uc29saWRhdGVkTWVudGlvblByb3RvEisKBHNwYW4YASABKAsyHS50dXJiby5yZXN1bHQudjEuU3BhbkZyYWdtZW50EhkKDGVudGl0eV9pbmRleBgCIAEoDUgAiAEBEhkKEWlzX3JlcHJlc2VudGF0aXZlGAMgASgIEhEKBHJvbGUYBCABKAlIAYgBAUIPCg1fZW50aXR5X2luZGV4QgcKBV9yb2xlImQKE0VudGl0eVJlbGF0aW9uUHJvdG8SDwoHc3ViamVjdBgBIAEoDRIOCgZvYmplY3QYAiABKA0SDAoEa2luZBgDIAEoCRINCgVzY29yZRgEIAEoAhIPCgdzdXBwb3J0GAUgASgNIuwDChhEb2N1bWVudEVudGl0eUFubm90YXRpb24SFgoOY2Fub25pY2FsX25hbWUYASABKAkSDAoEa2luZBgCIAEoCRISCgVrYl9pZBgDIAEoCUgAiAEBEhUKCGxhdGl0dWRlGAQgASgCSAGIAQESFgoJbG9uZ2l0dWRlGAUgASgCSAKIAQESGQoMY291bnRyeV9jb2RlGAYgASgJSAOIAQESOwoIbWVudGlvbnMYByADKAsyKS50dXJiby5yZXN1bHQudjEuQ29uc29saWRhdGVkTWVudGlvblByb3RvEhUKDXN1YmplY3RfY291bnQYCCABKA0SFAoMb2JqZWN0X2NvdW50GAkgASgNEhYKDm1vZGlmaWVyX2NvdW50GAogASgNEhMKC290aGVyX2NvdW50GAsgASgNEhYKDm1lYW5fc2VudGltZW50GAwgASgCEhoKEnNlbnRpbWVudF9tZW50aW9ucxgNIAEoDRISCgpjb25maWRlbmNlGA4gASgCEjcKCXJlbGF0aW9ucxgPIAMoCzIkLnR1cmJvLnJlc3VsdC52MS5FbnRpdHlSZWxhdGlvblByb3RvQggKBl9rYl9pZEILCglfbGF0aXR1ZGVCDAoKX2xvbmdpdHVkZUIPCg1fY291bnRyeV9jb2RlIpcBChJFdmVudEFyZ3VtZW50UHJvdG8SKwoEc3BhbhgBIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5TcGFuRnJhZ21lbnQSDAoEcm9sZRgCIAEoCRIWCg5hbmNob3JlZF9sYXllchgDIAEoCRIbCg5hbmNob3JlZF9pbmRleBgEIAEoDUgAiAEBQhEKD19hbmNob3JlZF9pbmRleCKjAQoPRXZlbnRBbm5vdGF0aW9uEhoKDXRyaWdnZXJfbGVtbWEYASABKAlIAIgBARIWCg5zZW50ZW5jZV9pbmRleBgCIAEoDRISCgpjb25maWRlbmNlGAMgASgCEjYKCWFyZ3VtZW50cxgEIAMoCzIjLnR1cmJvLnJlc3VsdC52MS5FdmVudEFyZ3VtZW50UHJvdG9CEAoOX3RyaWdnZXJfbGVtbWEiRAoNRWR1QW5ub3RhdGlvbhINCgVpbmRleBgBIAEoDRIWCg5zZW50ZW5jZV9pbmRleBgCIAEoDRIMCgR0ZXh0GAMgASgJIqYBChRTcGFyc2VUZXJtRW50cnlQcm90bxIPCgd0ZXJtX2lkGAEgASgNEg4KBndlaWdodBgCIAEoAhIwCgRzcGFuGAMgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudEgAiAEBEjIKC29jY3VycmVuY2VzGAQgAygLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudEIHCgVfc3BhbiJlChVTcGFyc2VUZXJtc0Fubm90YXRpb24SFAoMdGFyZ2V0X2luZGV4GAEgASgNEjYKB2VudHJpZXMYAiADKAsyJS50dXJiby5yZXN1bHQudjEuU3BhcnNlVGVybUVudHJ5UHJvdG8ijwEKFUFsdGVybmF0aXZlQW5ub3RhdGlvbhIQCghwb3NpdGlvbhgBIAEoDRIXCg9wb3NpdGlvbl9sZW5ndGgYAiABKA0SEQoJbW9kZV9iaXRzGAMgASgNEhIKBXNjb3JlGAQgASgBSACIAQESEQoEZm9ybRgFIAEoCUgBiAEBQggKBl9zY29yZUIHCgVfZm9ybSJyChNOQmVzdENhbmRpZGF0ZVByb3RvEgwKBHJhbmsYASABKA0SFgoOc2VxdWVuY2Vfc2NvcmUYAiABKAESCwoDdGFnGAMgASgJEhgKC3Byb2JhYmlsaXR5GAQgASgBSACIAQFCDgoMX3Byb2JhYmlsaXR5InEKD05CZXN0QW5ub3RhdGlvbhIUCgx0YXJnZXRfaW5kZXgYASABKA0SDgoGdGFnc2V0GAIgASgJEjgKCmNhbmRpZGF0ZXMYAyADKAsyJC50dXJiby5yZXN1bHQudjEuTkJlc3RDYW5kaWRhdGVQcm90byKTAQoTVGFibGVDZWxsQW5ub3RhdGlvbhIQCgh0YWJsZV9pZBgBIAEoDRIRCglyb3dfaW5kZXgYAiABKA0SEQoJY29sX2luZGV4GAMgASgNEhAKCHJvd19zcGFuGAQgASgNEhAKCGNvbF9zcGFuGAUgASgNEhEKCWlzX2hlYWRlchgGIAEoCBINCgVhbGlnbhgHIAEoCSLZAQoWS2V5VmFsdWVQYWlyQW5ub3RhdGlvbhIvCghrZXlfc3BhbhgBIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5TcGFuRnJhZ21lbnQSNQoOc2VwYXJhdG9yX3NwYW4YAiABKAsyHS50dXJiby5yZXN1bHQudjEuU3BhbkZyYWdtZW50EjEKCnZhbHVlX3NwYW4YAyABKAsyHS50dXJiby5yZXN1bHQudjEuU3BhbkZyYWdtZW50EhAKCGtleV90ZXh0GAQgASgJEhIKCnZhbHVlX3RleHQYBSABKAkisQEKGEVudGl0eVNhbGllbmNlQW5ub3RhdGlvbhIUCgxlbnRpdHlfaW5kZXgYASABKA0SEAoIc2FsaWVuY2UYAiABKAISDAoEcmFuaxgDIAEoDRIWCg5wb3NpdGlvbl9zY29yZRgEIAEoAhIUCgxzeW50YXhfc2NvcmUYBSABKAISFwoPZnJlcXVlbmN5X3Njb3JlGAYgASgCEhgKEGNlbnRyYWxpdHlfc2NvcmUYByABKAIiqAEKGkNodW5rVGFibGVIZWFkZXJBbm5vdGF0aW9uEhMKC2NodW5rX2luZGV4GAEgASgNEhAKCHRhYmxlX2lkGAIgASgNEhcKD2lzX2NvbnRpbnVhdGlvbhgDIAEoCBIzCgxoZWFkZXJfY2VsbHMYBCADKAsyHS50dXJiby5yZXN1bHQudjEuU3BhbkZyYWdtZW50EhUKDWhlYWRlcl9wcmVmaXgYBSABKAkigAEKFVJlY29yZFZhbHVlQW5ub3RhdGlvbhIUCgxyZWNvcmRfaW5kZXgYASABKA0SEwoLcmVjb3JkX3R5cGUYAiABKAkSEAoIaW5zdGFuY2UYAyABKA0SDAoEcGF0aBgEIAEoCRINCgVzY29yZRgFIAEoARINCgVsb2dpdBgGIAEoASL9BAoYUXVhbGl0eU1ldHJpY3NBbm5vdGF0aW9uEhwKD2NvbXBvc2l0ZV9zY29yZRgBIAEoAkgAiAEBEiMKFm10bGRfbGV4aWNhbF9kaXZlcnNpdHkYAiABKAJIAYgBARIhChRmbGVzY2hfa2luY2FpZF9ncmFkZRgDIAEoAkgCiAEBEigKG2F1dG9tYXRlZF9yZWFkYWJpbGl0eV9pbmRleBgEIAEoAkgDiAEBEh8KEmNvbGVtYW5fbGlhdV9pbmRleBgFIAEoAkgEiAEBEiAKE2luZm9ybWF0aW9uX2RlbnNpdHkYBiABKAJIBYgBARIYCgthbHBoYV9yYXRpbxgHIAEoAkgGiAEBEhgKC2RpZ2l0X3JhdGlvGAggASgCSAeIAQESGQoMc3ltYm9sX3JhdGlvGAkgASgCSAiIAQESHQoQd2hpdGVzcGFjZV9yYXRpbxgKIAEoAkgJiAEBEh4KEXJlcGxhY2VtZW50X3JhdGlvGAsgASgCSAqIAQESDQoFZmxhZ3MYDCABKA1CEgoQX2NvbXBvc2l0ZV9zY29yZUIZChdfbXRsZF9sZXhpY2FsX2RpdmVyc2l0eUIXChVfZmxlc2NoX2tpbmNhaWRfZ3JhZGVCHgocX2F1dG9tYXRlZF9yZWFkYWJpbGl0eV9pbmRleEIVChNfY29sZW1hbl9saWF1X2luZGV4QhYKFF9pbmZvcm1hdGlvbl9kZW5zaXR5Qg4KDF9hbHBoYV9yYXRpb0IOCgxfZGlnaXRfcmF0aW9CDwoNX3N5bWJvbF9yYXRpb0ITChFfd2hpdGVzcGFjZV9yYXRpb0IUChJfcmVwbGFjZW1lbnRfcmF0aW8iTgoPRXZpZGVuY2VTdXBwb3J0Eg0KBWxheWVyGAEgASgJEhAKCGluc3RhbmNlGAIgASgJEg0KBXN0YXJ0GAMgASgNEgsKA2VuZBgEIAEoDSLrAQoSRXZpZGVuY2VBbm5vdGF0aW9uEhQKDHRhcmdldF9pbmRleBgBIAEoDRIWCglyYXdfc2NvcmUYAiABKAFIAIgBARIjChZjYWxpYnJhdGVkX3Byb2JhYmlsaXR5GAMgASgBSAGIAQESFwoKYWJzdGVudGlvbhgEIAEoCUgCiAEBEjEKB3N1cHBvcnQYBSADKAsyIC50dXJiby5yZXN1bHQudjEuRXZpZGVuY2VTdXBwb3J0QgwKCl9yYXdfc2NvcmVCGQoXX2NhbGlicmF0ZWRfcHJvYmFiaWxpdHlCDQoLX2Fic3RlbnRpb24iwAIKE1F1b3RhdGlvbkFubm90YXRpb24SPAoEa2luZBgBIAEoDjIpLnR1cmJvLnJlc3VsdC52MS5RdW90YXRpb25Bbm5vdGF0aW9uLktpbmRIAIgBARIXCgpjb25maWRlbmNlGAIgASgCSAGIAQESLgoHc3BlYWtlchgDIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5TcGFuRnJhZ21lbnQSLwoIY3VlX3ZlcmIYBCABKAsyHS50dXJiby5yZXN1bHQudjEuU3BhbkZyYWdtZW50EhoKDXNwZWFrZXJfY2hhaW4YBSABKA1IAogBASIrCgRLaW5kEgoKBkRJUkVDVBAAEgwKCElORElSRUNUEAESCQoFTUlYRUQQAkIHCgVfa2luZEINCgtfY29uZmlkZW5jZUIQCg5fc3BlYWtlcl9jaGFpbiLwJwoPVHlwZWRBbm5vdGF0aW9uEiwKBXNwYW5zGAEgAygLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudBIYCgtwcm9iYWJpbGl0eRgCIAEoAUgBiAEBEhYKCW1vZGVsX3JlZhgDIAEoCUgCiAEBEjcKCHNlbnRlbmNlGAogASgLMiMudHVyYm8ucmVzdWx0LnYxLlNlbnRlbmNlQW5ub3RhdGlvbkgAEjEKBXRva2VuGAsgASgLMiAudHVyYm8ucmVzdWx0LnYxLlRva2VuQW5ub3RhdGlvbkgAEi8KBHdvcmQYDCABKAsyHy50dXJiby5yZXN1bHQudjEuV29yZEFubm90YXRpb25IABI1CgdzdWJ3b3JkGA0gASgLMiIudHVyYm8ucmVzdWx0LnYxLlN1YndvcmRBbm5vdGF0aW9uSAASNAoHcG9zX3RhZxgOIAEoCzIhLnR1cmJvLnJlc3VsdC52MS5Qb3NUYWdBbm5vdGF0aW9uSAASQgoObW9ycGhfZmVhdHVyZXMYDyABKAsyKC50dXJiby5yZXN1bHQudjEuTW9ycGhGZWF0dXJlc0Fubm90YXRpb25IABIxCgVsZW1tYRgQIAEoCzIgLnR1cmJvLnJlc3VsdC52MS5MZW1tYUFubm90YXRpb25IABIvCgRzdGVtGBEgASgLMh8udHVyYm8ucmVzdWx0LnYxLlN0ZW1Bbm5vdGF0aW9uSAASLwoEdGVybRgSIAEoCzIfLnR1cmJvLnJlc3VsdC52MS5UZXJtQW5ub3RhdGlvbkgAEjgKCWNodW5rX3RhZxgTIAEoCzIjLnR1cmJvLnJlc3VsdC52MS5DaHVua1RhZ0Fubm90YXRpb25IABI+CgxwaHJhc2VfY2h1bmsYFCABKAsyJi50dXJiby5yZXN1bHQudjEuUGhyYXNlQ2h1bmtBbm5vdGF0aW9uSAASMwoGcGhyYXNlGBUgASgLMiEudHVyYm8ucmVzdWx0LnYxLlBocmFzZUFubm90YXRpb25IABI7CgpkZXBlbmRlbmN5GBYgASgLMiUudHVyYm8ucmVzdWx0LnYxLkRlcGVuZGVuY3lBbm5vdGF0aW9uSAASMwoGZW50aXR5GBcgASgLMiEudHVyYm8ucmVzdWx0LnYxLkVudGl0eUFubm90YXRpb25IABI3CghyZWxhdGlvbhgYIAEoCzIjLnR1cmJvLnJlc3VsdC52MS5SZWxhdGlvbkFubm90YXRpb25IABJACg1jb3JlZl9tZW50aW9uGBkgASgLMicudHVyYm8ucmVzdWx0LnYxLkNvcmVmTWVudGlvbkFubm90YXRpb25IABI8Cgtjb3JlZl9jaGFpbhgaIAEoCzIlLnR1cmJvLnJlc3VsdC52MS5Db3JlZkNoYWluQW5ub3RhdGlvbkgAEjUKB3NwZWFrZXIYGyABKAsyIi50dXJiby5yZXN1bHQudjEuU3BlYWtlckFubm90YXRpb25IABJACg13b3JkbmV0X3NlbnNlGBwgASgLMicudHVyYm8ucmVzdWx0LnYxLldvcmROZXRTZW5zZUFubm90YXRpb25IABJIChFsZXhpY2FsX2V4cGFuc2lvbhgdIAEoCzIrLnR1cmJvLnJlc3VsdC52MS5MZXhpY2FsRXhwYW5zaW9uQW5ub3RhdGlvbkgAEkAKDWdsb3NzYXJ5X3Rlcm0YHiABKAsyJy50dXJiby5yZXN1bHQudjEuR2xvc3NhcnlUZXJtQW5ub3RhdGlvbkgAEjMKBm51bWJlchgfIAEoCzIhLnR1cmJvLnJlc3VsdC52MS5OdW1iZXJBbm5vdGF0aW9uSAASNQoHb3JkaW5hbBggIAEoCzIiLnR1cmJvLnJlc3VsdC52MS5PcmRpbmFsQW5ub3RhdGlvbkgAEjEKBW1vbmV5GCEgASgLMiAudHVyYm8ucmVzdWx0LnYxLk1vbmV5QW5ub3RhdGlvbkgAEjcKCHF1YW50aXR5GCIgASgLMiMudHVyYm8ucmVzdWx0LnYxLlF1YW50aXR5QW5ub3RhdGlvbkgAEjcKCHRlbXBvcmFsGCMgASgLMiMudHVyYm8ucmVzdWx0LnYxLlRlbXBvcmFsQW5ub3RhdGlvbkgAEi8KBHRpbWUYJCABKAsyHy50dXJiby5yZXN1bHQudjEuVGltZUFubm90YXRpb25IABJACg1kb2N1bWVudF9kYXRlGCUgASgLMicudHVyYm8ucmVzdWx0LnYxLkRvY3VtZW50RGF0ZUFubm90YXRpb25IABIxCgVwbGFjZRgmIAEoCzIgLnR1cmJvLnJlc3VsdC52MS5QbGFjZVJlc29sdXRpb25IABI4Cgtjb250YWlubWVudBgnIAEoCzIhLnR1cmJvLnJlc3VsdC52MS5Db250YWlubWVudENoYWluSAASMgoLcmVnaW9uX3ZvdGUYKCABKAsyGy50dXJiby5yZXN1bHQudjEuUmVnaW9uVm90ZUgAEjcKCGxhbmd1YWdlGCkgASgLMiMudHVyYm8ucmVzdWx0LnYxLkxhbmd1YWdlQW5ub3RhdGlvbkgAEkAKDWxhbmd1YWdlX3NwYW4YKiABKAsyJy50dXJiby5yZXN1bHQudjEuTGFuZ3VhZ2VTcGFuQW5ub3RhdGlvbkgAEjcKCGNhdGVnb3J5GCsgASgLMiMudHVyYm8ucmVzdWx0LnYxLkNhdGVnb3J5QW5ub3RhdGlvbkgAEjkKCXNlbnRpbWVudBgsIAEoCzIkLnR1cmJvLnJlc3VsdC52MS5TZW50aW1lbnRBbm5vdGF0aW9uSAASNQoHZW1vdGlvbhgtIAEoCzIiLnR1cmJvLnJlc3VsdC52MS5FbW90aW9uQW5ub3RhdGlvbkgAEjMKBmFzcGVjdBguIAEoCzIhLnR1cmJvLnJlc3VsdC52MS5Bc3BlY3RBbm5vdGF0aW9uSAASNQoHb3BpbmlvbhgvIAEoCzIiLnR1cmJvLnJlc3VsdC52MS5PcGluaW9uQW5ub3RhdGlvbkgAEjsKCmluY2l2aWxpdHkYMCABKAsyJS50dXJiby5yZXN1bHQudjEuSW5jaXZpbGl0eUFubm90YXRpb25IABI7CgpwcmFnbWF0aWNzGDEgASgLMiUudHVyYm8ucmVzdWx0LnYxLlByYWdtYXRpY3NBbm5vdGF0aW9uSAASKgoDcGlpGDIgASgLMhsudHVyYm8ucmVzdWx0LnYxLlBpaU1lbnRpb25IABIrCgVub2lzZRgzIAEoCzIaLnR1cmJvLnJlc3VsdC52MS5Ob2lzZVNwYW5IABIxCghhcnRpZmFjdBg0IAEoCzIdLnR1cmJvLnJlc3VsdC52MS5UZXh0QXJ0aWZhY3RIABIvCgVhc3NldBg1IAEoCzIeLnR1cmJvLnJlc3VsdC52MS5FbWJlZGRlZEFzc2V0SAASOQoKY29ycmVjdGlvbhg2IAEoCzIjLnR1cmJvLnJlc3VsdC52MS5TcGVsbGluZ0NvcnJlY3Rpb25IABI9CgtwbGFjZWhvbGRlchg3IAEoCzImLnR1cmJvLnJlc3VsdC52MS5QbGFjZWhvbGRlckFubm90YXRpb25IABIxCgVlbW9qaRg4IAEoCzIgLnR1cmJvLnJlc3VsdC52MS5FbW9qaUFubm90YXRpb25IABJCCg5wcm90ZWN0ZWRfc3Bhbhg5IAEoCzIoLnR1cmJvLnJlc3VsdC52MS5Qcm90ZWN0ZWRTcGFuQW5ub3RhdGlvbkgAEj8KDGFiYnJldmlhdGlvbhg6IAEoCzInLnR1cmJvLnJlc3VsdC52MS5BYmJyZXZpYXRpb25Bbm5vdGF0aW9uSAASNgoKcnVsZV9tYXRjaBg7IAEoCzIgLnR1cmJvLnJlc3VsdC52MS5NYXRjaEFubm90YXRpb25IABI9Cgtib2lsZXJwbGF0ZRg8IAEoCzImLnR1cmJvLnJlc3VsdC52MS5Cb2lsZXJwbGF0ZUFubm90YXRpb25IABI7Cgpjb25maWRlbmNlGD0gASgLMiUudHVyYm8ucmVzdWx0LnYxLkNvbmZpZGVuY2VBbm5vdGF0aW9uSAASQAoNc3RvcHdvcmRfZmxhZxg+IAEoCzInLnR1cmJvLnJlc3VsdC52MS5TdG9wd29yZEZsYWdBbm5vdGF0aW9uSAASOAoJd29yZF90eXBlGD8gASgLMiMudHVyYm8ucmVzdWx0LnYxLldvcmRUeXBlQW5ub3RhdGlvbkgAEjUKB3JlYWRpbmcYQCABKAsyIi50dXJiby5yZXN1bHQudjEuUmVhZGluZ0Fubm90YXRpb25IABI1CgdrZXl3b3JkGEEgASgLMiIudHVyYm8ucmVzdWx0LnYxLktleXdvcmRBbm5vdGF0aW9uSAASRAoPc291cmNlX2VuY29kaW5nGEIgASgLMikudHVyYm8ucmVzdWx0LnYxLlNvdXJjZUVuY29kaW5nQW5ub3RhdGlvbkgAEjMKBnZlY3RvchhDIAEoCzIhLnR1cmJvLnJlc3VsdC52MS5WZWN0b3JBbm5vdGF0aW9uSAASMQoFY2h1bmsYRCABKAsyIC50dXJiby5yZXN1bHQudjEuQ2h1bmtBbm5vdGF0aW9uSAASNgoKbGV2ZWxfdW5pdBhFIAEoCzIgLnR1cmJvLnJlc3VsdC52MS5MZXZlbEFubm90YXRpb25IABI3Cghib3VuZGFyeRhGIAEoCzIjLnR1cmJvLnJlc3VsdC52MS5Cb3VuZGFyeUFubm90YXRpb25IABI5CglzdHJ1Y3R1cmUYRyABKAsyJC50dXJiby5yZXN1bHQudjEuU3RydWN0dXJlQW5ub3RhdGlvbkgAEjMKBmFuc3dlchhIIAEoCzIhLnR1cmJvLnJlc3VsdC52MS5BbnN3ZXJBbm5vdGF0aW9uSAASLwoEZWRnZRhJIAEoCzIfLnR1cmJvLnJlc3VsdC52MS5FZGdlQW5ub3RhdGlvbkgAEjEKBWdyb3VwGEogASgLMiAudHVyYm8ucmVzdWx0LnYxLkdyb3VwQW5ub3RhdGlvbkgAEkAKDWxpbmtlZF9lbnRpdHkYSyABKAsyJy50dXJiby5yZXN1bHQudjEuTGlua2VkRW50aXR5QW5ub3RhdGlvbkgAEjsKCmNhbGlicmF0ZWQYTCABKAsyJS50dXJiby5yZXN1bHQudjEuQ2FsaWJyYXRlZEFubm90YXRpb25IABJLChN0b2tlbl92ZWN0b3Jfd2luZG93GE0gASgLMiwudHVyYm8ucmVzdWx0LnYxLlRva2VuVmVjdG9yV2luZG93QW5ub3RhdGlvbkgAEjoKCnV0ZjE2X3NwYW4YTiABKAsyJC50dXJiby5yZXN1bHQudjEuVXRmMTZTcGFuQW5ub3RhdGlvbkgAEjcKCG1vZGFsaXR5GE8gASgLMiMudHVyYm8ucmVzdWx0LnYxLk1vZGFsaXR5QW5ub3RhdGlvbkgAEkQKD2RvY3VtZW50X2VudGl0eRhQIAEoCzIpLnR1cmJvLnJlc3VsdC52MS5Eb2N1bWVudEVudGl0eUFubm90YXRpb25IABIxCgVldmVudBhRIAEoCzIgLnR1cmJvLnJlc3VsdC52MS5FdmVudEFubm90YXRpb25IABItCgNlZHUYUiABKAsyHi50dXJiby5yZXN1bHQudjEuRWR1QW5ub3RhdGlvbkgAEj4KDHNwYXJzZV90ZXJtcxhTIAEoCzImLnR1cmJvLnJlc3VsdC52MS5TcGFyc2VUZXJtc0Fubm90YXRpb25IABI9CgthbHRlcm5hdGl2ZRhUIAEoCzImLnR1cmJvLnJlc3VsdC52MS5BbHRlcm5hdGl2ZUFubm90YXRpb25IABIxCgVuYmVzdBhVIAEoCzIgLnR1cmJvLnJlc3VsdC52MS5OQmVzdEFubm90YXRpb25IABI6Cgp0YWJsZV9jZWxsGFYgASgLMiQudHVyYm8ucmVzdWx0LnYxLlRhYmxlQ2VsbEFubm90YXRpb25IABJBCg5rZXlfdmFsdWVfcGFpchhXIAEoCzInLnR1cmJvLnJlc3VsdC52MS5LZXlWYWx1ZVBhaXJBbm5vdGF0aW9uSAASPgoMcmVjb3JkX3ZhbHVlGFggASgLMiYudHVyYm8ucmVzdWx0LnYxLlJlY29yZFZhbHVlQW5ub3RhdGlvbkgAEkQKD2VudGl0eV9zYWxpZW5jZRhZIAEoCzIpLnR1cmJvLnJlc3VsdC52MS5FbnRpdHlTYWxpZW5jZUFubm90YXRpb25IABJJChJjaHVua190YWJsZV9oZWFkZXIYWiABKAsyKy50dXJiby5yZXN1bHQudjEuQ2h1bmtUYWJsZUhlYWRlckFubm90YXRpb25IABJBCgx0ZXh0X3F1YWxpdHkYWyABKAsyKS50dXJiby5yZXN1bHQudjEuUXVhbGl0eU1ldHJpY3NBbm5vdGF0aW9uSAASQQoMc3Bhbl9xdWFsaXR5GFwgASgLMikudHVyYm8ucmVzdWx0LnYxLlF1YWxpdHlNZXRyaWNzQW5ub3RhdGlvbkgAEjcKCGV2aWRlbmNlGF0gASgLMiMudHVyYm8ucmVzdWx0LnYxLkV2aWRlbmNlQW5ub3RhdGlvbkgAEjkKCXF1b3RhdGlvbhheIAEoCzIkLnR1cmJvLnJlc3VsdC52MS5RdW90YXRpb25Bbm5vdGF0aW9uSABCBwoFdmFsdWVCDgoMX3Byb2JhYmlsaXR5QgwKCl9tb2RlbF9yZWYqVQoKT2Zmc2V0TW9kZRIUChBPRkZTRVRfTU9ERV9VVEY4EAASFQoRT0ZGU0VUX01PREVfVVRGMTYQARIaChZPRkZTRVRfTU9ERV9DT0RFX1BPSU5UEAIqaAoPTGF5ZXJXaXJlU3RhdHVzEhgKFExBWUVSX1dJUkVfU1RBVFVTX09LEAASHQoZTEFZRVJfV0lSRV9TVEFUVVNfU0tJUFBFRBABEhwKGExBWUVSX1dJUkVfU1RBVFVTX0ZBSUxFRBACKlUKCkFuY2hvcktpbmQSFAoQQU5DSE9SX0tJTkRfVEVYVBAAEhcKE0FOQ0hPUl9LSU5EX0FMSUdORUQQARIYChRBTkNIT1JfS0lORF9ET0NVTUVOVBACYgZwcm90bzM");
var LayerWireStatus = /* @__PURE__ */ ((LayerWireStatus2) => {
  LayerWireStatus2[LayerWireStatus2["OK"] = 0] = "OK";
  LayerWireStatus2[LayerWireStatus2["SKIPPED"] = 1] = "SKIPPED";
  LayerWireStatus2[LayerWireStatus2["FAILED"] = 2] = "FAILED";
  return LayerWireStatus2;
})(LayerWireStatus || {});

// demo/web/src/dom.ts
function $(id) {
  const e = document.getElementById(id);
  if (!e) throw new Error(`no element #${id}`);
  return e;
}
function input(id) {
  return $(id);
}
function el(tag, props = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === void 0) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "style") e.style.cssText = v;
    else e.setAttribute(k, v);
  }
  append(e, children);
  return e;
}
function append(e, children) {
  for (const c of children) {
    if (c === null || c === void 0 || c === false) continue;
    if (Array.isArray(c)) append(e, c);
    else e.append(c);
  }
}
function stat(value, key) {
  return el("div", { class: "stat" }, el("div", { class: "v", text: value }), el("div", { class: "k", text: key }));
}
function table(head, rows, cls = "") {
  return el(
    "table",
    { class: cls || void 0 },
    el("tr", {}, head.map((h) => el("th", { text: h }))),
    rows.map((r) => el("tr", {}, r.map((c) => el("td", {}, c))))
  );
}
function options(select4, values, labels) {
  const keep = select4.value;
  select4.replaceChildren(...values.map((v, i) => el("option", { value: v, text: labels ? labels[i] : v })));
  if (values.includes(keep)) select4.value = keep;
}
function say(box, text, kind = "") {
  box.textContent = text;
  box.className = kind ? `note ${kind}` : "note";
  box.hidden = text === "";
}
function size(n) {
  const v = Number(n);
  if (v < 1024) return `${v} B`;
  if (v < 1 << 20) return `${(v / 1024).toFixed(1)} KiB`;
  if (v < 1 << 30) return `${(v / (1 << 20)).toFixed(1)} MiB`;
  return `${(v / (1 << 30)).toFixed(2)} GiB`;
}
function download(name, data, type) {
  const url = URL.createObjectURL(new Blob(data, { type }));
  const a = el("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1e4);
}
function after(s, prefix2) {
  return s.startsWith(prefix2) ? s.slice(prefix2.length) : s;
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/code.js
var Code;
(function(Code2) {
  Code2[Code2["Canceled"] = 1] = "Canceled";
  Code2[Code2["Unknown"] = 2] = "Unknown";
  Code2[Code2["InvalidArgument"] = 3] = "InvalidArgument";
  Code2[Code2["DeadlineExceeded"] = 4] = "DeadlineExceeded";
  Code2[Code2["NotFound"] = 5] = "NotFound";
  Code2[Code2["AlreadyExists"] = 6] = "AlreadyExists";
  Code2[Code2["PermissionDenied"] = 7] = "PermissionDenied";
  Code2[Code2["ResourceExhausted"] = 8] = "ResourceExhausted";
  Code2[Code2["FailedPrecondition"] = 9] = "FailedPrecondition";
  Code2[Code2["Aborted"] = 10] = "Aborted";
  Code2[Code2["OutOfRange"] = 11] = "OutOfRange";
  Code2[Code2["Unimplemented"] = 12] = "Unimplemented";
  Code2[Code2["Internal"] = 13] = "Internal";
  Code2[Code2["Unavailable"] = 14] = "Unavailable";
  Code2[Code2["DataLoss"] = 15] = "DataLoss";
  Code2[Code2["Unauthenticated"] = 16] = "Unauthenticated";
})(Code || (Code = {}));

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/wkt/google/protobuf/any_pb.js
var file_google_protobuf_any = /* @__PURE__ */ fileDesc("Chlnb29nbGUvcHJvdG9idWYvYW55LnByb3RvEg9nb29nbGUucHJvdG9idWYiJgoDQW55EhAKCHR5cGVfdXJsGAEgASgJEg0KBXZhbHVlGAIgASgMQnYKE2NvbS5nb29nbGUucHJvdG9idWZCCEFueVByb3RvUAFaLGdvb2dsZS5nb2xhbmcub3JnL3Byb3RvYnVmL3R5cGVzL2tub3duL2FueXBiogIDR1BCqgIeR29vZ2xlLlByb3RvYnVmLldlbGxLbm93blR5cGVzYgZwcm90bzM");
var AnySchema = /* @__PURE__ */ messageDesc(file_google_protobuf_any, 0);

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/wkt/any.js
function anyPack(schema, message, into) {
  let ret = false;
  if (!into) {
    into = create(AnySchema);
    ret = true;
  }
  into.value = toBinary(schema, message);
  into.typeUrl = typeNameToUrl(message.$typeName);
  return ret ? into : void 0;
}
function anyIs(any, descOrTypeName) {
  if (any.typeUrl === "") {
    return false;
  }
  const want = typeof descOrTypeName == "string" ? descOrTypeName : descOrTypeName.typeName;
  const got = typeUrlToName(any.typeUrl);
  return want === got;
}
function anyUnpack(any, registryOrMessageDesc) {
  if (any.typeUrl === "") {
    return void 0;
  }
  const desc = registryOrMessageDesc.kind == "message" ? registryOrMessageDesc : registryOrMessageDesc.getMessage(typeUrlToName(any.typeUrl));
  if (!desc || !anyIs(any, desc)) {
    return void 0;
  }
  return fromBinary(desc, any.value);
}
function typeNameToUrl(name) {
  return `type.googleapis.com/${name}`;
}
function typeUrlToName(url) {
  const slash = url.lastIndexOf("/");
  const name = slash >= 0 ? url.substring(slash + 1) : url;
  if (!name.length) {
    throw new Error(`invalid type url: ${url}`);
  }
  return name;
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/wkt/google/protobuf/struct_pb.js
var file_google_protobuf_struct = /* @__PURE__ */ fileDesc("Chxnb29nbGUvcHJvdG9idWYvc3RydWN0LnByb3RvEg9nb29nbGUucHJvdG9idWYihAEKBlN0cnVjdBIzCgZmaWVsZHMYASADKAsyIy5nb29nbGUucHJvdG9idWYuU3RydWN0LkZpZWxkc0VudHJ5GkUKC0ZpZWxkc0VudHJ5EgsKA2tleRgBIAEoCRIlCgV2YWx1ZRgCIAEoCzIWLmdvb2dsZS5wcm90b2J1Zi5WYWx1ZToCOAEi6gEKBVZhbHVlEjAKCm51bGxfdmFsdWUYASABKA4yGi5nb29nbGUucHJvdG9idWYuTnVsbFZhbHVlSAASFgoMbnVtYmVyX3ZhbHVlGAIgASgBSAASFgoMc3RyaW5nX3ZhbHVlGAMgASgJSAASFAoKYm9vbF92YWx1ZRgEIAEoCEgAEi8KDHN0cnVjdF92YWx1ZRgFIAEoCzIXLmdvb2dsZS5wcm90b2J1Zi5TdHJ1Y3RIABIwCgpsaXN0X3ZhbHVlGAYgASgLMhouZ29vZ2xlLnByb3RvYnVmLkxpc3RWYWx1ZUgAQgYKBGtpbmQiMwoJTGlzdFZhbHVlEiYKBnZhbHVlcxgBIAMoCzIWLmdvb2dsZS5wcm90b2J1Zi5WYWx1ZSobCglOdWxsVmFsdWUSDgoKTlVMTF9WQUxVRRAAQn8KE2NvbS5nb29nbGUucHJvdG9idWZCC1N0cnVjdFByb3RvUAFaL2dvb2dsZS5nb2xhbmcub3JnL3Byb3RvYnVmL3R5cGVzL2tub3duL3N0cnVjdHBi+AEBogIDR1BCqgIeR29vZ2xlLlByb3RvYnVmLldlbGxLbm93blR5cGVzYgZwcm90bzM");
var StructSchema = /* @__PURE__ */ messageDesc(file_google_protobuf_struct, 0);
var ValueSchema = /* @__PURE__ */ messageDesc(file_google_protobuf_struct, 1);
var ListValueSchema = /* @__PURE__ */ messageDesc(file_google_protobuf_struct, 2);
var NullValue;
(function(NullValue2) {
  NullValue2[NullValue2["NULL_VALUE"] = 0] = "NULL_VALUE";
})(NullValue || (NullValue = {}));

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/extensions.js
function getExtension(message, extension, options2) {
  assertExtendee(extension, message);
  const ufs = filterUnknownFields(message.$unknown, extension);
  const [container, field, get] = createExtensionContainer(extension);
  const ctx = makeReadContext(options2);
  for (const uf of ufs) {
    readField(container, new BinaryReader(uf.data), field, uf.wireType, ctx);
  }
  return get();
}
function setExtension(message, extension, value) {
  var _a;
  assertExtendee(extension, message);
  const ufs = ((_a = message.$unknown) !== null && _a !== void 0 ? _a : []).filter((uf) => uf.no !== extension.number);
  const [container, field] = createExtensionContainer(extension, value);
  const writer = new BinaryWriter();
  writeField(writer, { writeUnknownFields: true }, container, field);
  const reader = new BinaryReader(writer.finish());
  while (reader.pos < reader.len) {
    const [no, wireType] = reader.tag();
    const data = reader.skip(wireType, no);
    ufs.push({ no, wireType, data });
  }
  message.$unknown = ufs;
}
function filterUnknownFields(unknownFields, extension) {
  if (unknownFields === void 0)
    return [];
  if (extension.fieldKind === "enum" || extension.fieldKind === "scalar") {
    for (let i = unknownFields.length - 1; i >= 0; --i) {
      if (unknownFields[i].no == extension.number) {
        return [unknownFields[i]];
      }
    }
    return [];
  }
  return unknownFields.filter((uf) => uf.no === extension.number);
}
function createExtensionContainer(extension, value) {
  const localName = extension.typeName;
  const field = Object.assign(Object.assign({}, extension), { kind: "field", parent: extension.extendee, localName });
  const desc = Object.assign(Object.assign({}, extension.extendee), { fields: [field], members: [field], oneofs: [] });
  const container = create(desc, value !== void 0 ? { [localName]: value } : void 0);
  return [
    reflect(desc, container),
    field,
    () => {
      const value2 = container[localName];
      if (value2 === void 0) {
        const desc2 = extension.message;
        if (isWrapperDesc(desc2)) {
          return scalarZeroValue(desc2.fields[0].scalar, desc2.fields[0].longAsString);
        }
        return create(desc2);
      }
      return value2;
    }
  ];
}
function assertExtendee(extension, message) {
  if (extension.extendee.typeName != message.$typeName) {
    throw new Error(`extension ${extension.typeName} can only be applied to message ${extension.extendee.typeName}`);
  }
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/wkt/json.js
var timestampMsMin = /* @__PURE__ */ Date.parse("0001-01-01T00:00:00Z");
var timestampMsMax = /* @__PURE__ */ Date.parse("9999-12-31T23:59:59Z");
var durationSecondsMin = -315576e6;
var durationSecondsMax = 315576e6;

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/to-json.js
var LEGACY_REQUIRED3 = 3;
var IMPLICIT5 = 2;
var jsonWriteDefaults = {
  alwaysEmitImplicit: false,
  enumAsInteger: false,
  useProtoFieldName: false
};
function makeWriteOptions2(options2) {
  return options2 ? Object.assign(Object.assign({}, jsonWriteDefaults), options2) : jsonWriteDefaults;
}
function toJson(schema, message, options2) {
  return compiledWriter2(schema)(makeWriteOptions2(options2), message);
}
function toJsonString(schema, message, options2) {
  var _a;
  const jsonValue = toJson(schema, message, options2);
  return JSON.stringify(jsonValue, null, (_a = options2 === null || options2 === void 0 ? void 0 : options2.prettySpaces) !== null && _a !== void 0 ? _a : 0);
}
var compiledWriters2 = /* @__PURE__ */ new WeakMap();
function compiledWriter2(desc) {
  let compiled = compiledWriters2.get(desc);
  if (compiled === void 0) {
    compiled = compileMessage3(desc);
  }
  return compiled;
}
function compileMessage3(desc) {
  const typeName = desc.typeName;
  const writeWkt = compileWkt(desc);
  if (writeWkt !== void 0) {
    const foreignField2 = desc.fields[0];
    const compiledWriter4 = (opts, message) => {
      if (message.$typeName !== typeName && foreignField2 !== void 0) {
        throw new FieldError(foreignField2, `cannot use ${foreignField2} with message ${message.$typeName}`, "ForeignFieldError");
      }
      return writeWkt(opts, message);
    };
    compiledWriters2.set(desc, compiledWriter4);
    return compiledWriter4;
  }
  const sortedFields = desc.fields.concat().sort((a, b) => a.number - b.number);
  const foreignField = sortedFields[0];
  const fieldWriters = [];
  const compiledWriter3 = (opts, message) => {
    if (message.$typeName !== typeName && foreignField !== void 0) {
      throw new FieldError(foreignField, `cannot use ${foreignField} with message ${message.$typeName}`, "ForeignFieldError");
    }
    const json = {};
    for (let i = 0; i < fieldWriters.length; i++) {
      fieldWriters[i](opts, message, json);
    }
    if (opts.registry) {
      writeExtensions(json, opts, opts.registry, message, desc);
    }
    return json;
  };
  compiledWriters2.set(desc, compiledWriter3);
  for (const field of sortedFields) {
    const writeField2 = compileField2(field);
    fieldWriters.push(field.jsonName === "__proto__" || field.name === "__proto__" ? writeOwnKeys(writeField2) : writeField2);
  }
  return compiledWriter3;
}
function writeOwnKeys(writeField2) {
  return (opts, message, json) => {
    const tmp = /* @__PURE__ */ Object.create(null);
    writeField2(opts, message, tmp);
    for (const key of Object.keys(tmp)) {
      setOwn(json, key, tmp[key]);
    }
  };
}
function compileWkt(desc) {
  if (!desc.typeName.startsWith("google.protobuf.")) {
    return void 0;
  }
  switch (desc.typeName) {
    case "google.protobuf.Any":
      return (opts, message) => anyToJson(message, opts);
    case "google.protobuf.Timestamp":
      return (opts, message) => timestampToJson(message);
    case "google.protobuf.Duration":
      return (opts, message) => durationToJson(message);
    case "google.protobuf.FieldMask":
      return (opts, message) => fieldMaskToJson(message);
    case "google.protobuf.Struct":
      return (opts, message) => structToJson(message);
    case "google.protobuf.Value":
      return (opts, message) => valueToJson(message);
    case "google.protobuf.ListValue":
      return (opts, message) => listValueToJson(message);
    default:
      if (isWrapperDesc(desc)) {
        const valueField = desc.fields[0];
        const localName = valueField.localName;
        const zero = scalarZeroValue(valueField.scalar, false);
        const writeScalar = compileScalarValue2(valueField);
        return (opts, message) => {
          const value = message[localName];
          return writeScalar(opts, value === void 0 ? zero : value);
        };
      }
      return void 0;
  }
}
function compileField2(field) {
  switch (field.fieldKind) {
    case "scalar":
    case "enum":
    case "message":
      return compileSingularField2(field);
    case "list":
    case "map": {
      const writeValue = field.fieldKind == "list" ? compileListValue(field) : compileMapValue(field);
      const protoName = field.name;
      const jsonKey = field.jsonName;
      const localName = field.localName;
      return (opts, message, json) => {
        const value = writeValue(opts, message[localName]);
        if (value !== void 0) {
          json[opts.useProtoFieldName ? protoName : jsonKey] = value;
        }
      };
    }
  }
}
function compileSingularField2(field) {
  const writeValue = compileSingularValue2(field);
  const protoName = field.name;
  const jsonKey = field.jsonName;
  const localName = field.localName;
  if (field.oneof) {
    const oneofLocalName = field.oneof.localName;
    return (opts, message, json) => {
      const oneof = message[oneofLocalName];
      if (oneof.case === localName) {
        json[opts.useProtoFieldName ? protoName : jsonKey] = writeValue(opts, oneof.value);
      }
    };
  }
  if (field.presence != IMPLICIT5) {
    const requiredError = field.presence == LEGACY_REQUIRED3 ? `cannot encode ${field} to JSON: required field not set` : void 0;
    return (opts, message, json) => {
      const value = message[localName];
      if (value !== void 0 && Object.prototype.hasOwnProperty.call(message, localName)) {
        json[opts.useProtoFieldName ? protoName : jsonKey] = writeValue(opts, value);
      } else if (requiredError !== void 0) {
        throw new Error(requiredError);
      }
    };
  }
  if (field.fieldKind == "enum") {
    const zero = field.enum.values[0].number;
    return (opts, message, json) => {
      const value = message[localName];
      if (value !== zero || opts.alwaysEmitImplicit) {
        json[opts.useProtoFieldName ? protoName : jsonKey] = writeValue(opts, value);
      }
    };
  }
  switch (field.scalar) {
    case ScalarType.BOOL:
      return (opts, message, json) => {
        const value = message[localName];
        if (value !== false || opts.alwaysEmitImplicit) {
          json[opts.useProtoFieldName ? protoName : jsonKey] = writeValue(opts, value);
        }
      };
    case ScalarType.STRING:
      return (opts, message, json) => {
        const value = message[localName];
        if (value !== "" || opts.alwaysEmitImplicit) {
          json[opts.useProtoFieldName ? protoName : jsonKey] = writeValue(opts, value);
        }
      };
    case ScalarType.BYTES:
      return (opts, message, json) => {
        const value = message[localName];
        if (!(value instanceof Uint8Array) || value.byteLength > 0 || opts.alwaysEmitImplicit) {
          json[opts.useProtoFieldName ? protoName : jsonKey] = writeValue(opts, value);
        }
      };
    case ScalarType.DOUBLE:
    case ScalarType.FLOAT:
      return (opts, message, json) => {
        const value = message[localName];
        if (!Object.is(value, 0) || opts.alwaysEmitImplicit) {
          json[opts.useProtoFieldName ? protoName : jsonKey] = writeValue(opts, value);
        }
      };
    default:
      return (opts, message, json) => {
        const value = message[localName];
        if (value != 0 || opts.alwaysEmitImplicit) {
          json[opts.useProtoFieldName ? protoName : jsonKey] = writeValue(opts, value);
        }
      };
  }
}
function compileFieldValue(field) {
  switch (field.fieldKind) {
    case "scalar":
    case "enum":
    case "message":
      return compileSingularValue2(field);
    case "list":
      return compileListValue(field);
    case "map":
      return compileMapValue(field);
  }
}
function compileSingularValue2(field) {
  switch (field.fieldKind) {
    case "scalar":
      return compileScalarValue2(field);
    case "enum":
      return compileEnumValue(field);
    case "message":
      return compileMessageValue(field);
  }
}
function compileMessageValue(field) {
  const { toMessage } = localMessageMapper(field);
  const writeMessage = compiledWriter2(field.message);
  return (opts, value) => writeMessage(opts, toMessage(value));
}
function compileListValue(field) {
  const writeItem = compileListItemValue(field);
  return (opts, value) => {
    const items = value;
    if (items.length == 0 && !opts.alwaysEmitImplicit) {
      return void 0;
    }
    const jsonArray = [];
    for (let i = 0; i < items.length; i++) {
      jsonArray.push(writeItem(opts, items[i]));
    }
    return jsonArray;
  };
}
function compileListItemValue(field) {
  switch (field.listKind) {
    case "scalar":
      return compileScalarValue2(field);
    case "enum":
      return compileEnumValue(field);
    case "message":
      return compileMessageValue(field);
  }
}
function compileMapValue(field) {
  const writeMapValue = compileMapEntryValue(field);
  return (opts, value) => {
    const record = value;
    const keys = Object.keys(record);
    if (keys.length == 0 && !opts.alwaysEmitImplicit) {
      return void 0;
    }
    const jsonObject = {};
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      setOwn(jsonObject, key, writeMapValue(opts, record[key]));
    }
    return jsonObject;
  };
}
function compileMapEntryValue(field) {
  switch (field.mapKind) {
    case "scalar":
      return compileScalarValue2(field);
    case "enum":
      return compileEnumValue(field);
    case "message":
      return compileMessageValue(field);
  }
}
function compileEnumValue(field) {
  const desc = field.enum;
  if (desc.typeName == "google.protobuf.NullValue") {
    return (opts, value) => {
      if (typeof value != "number") {
        throw errorEnumValue(desc, value);
      }
      return null;
    };
  }
  return (opts, value) => {
    var _a, _b;
    if (typeof value != "number") {
      throw errorEnumValue(desc, value);
    }
    if (opts.enumAsInteger) {
      return value;
    }
    return (_b = (_a = desc.value[value]) === null || _a === void 0 ? void 0 : _a.jsonName) !== null && _b !== void 0 ? _b : value;
  };
}
function errorEnumValue(desc, value) {
  return new Error(`cannot encode ${desc} to JSON: expected number, got ${formatVal(value)}`);
}
function compileScalarValue2(field) {
  switch (field.scalar) {
    // int32, fixed32, uint32: JSON value will be a decimal number. Either numbers or strings are accepted.
    case ScalarType.INT32:
    case ScalarType.SFIXED32:
    case ScalarType.SINT32:
    case ScalarType.FIXED32:
    case ScalarType.UINT32:
      return (opts, value) => {
        if (typeof value != "number") {
          throw errorScalarValue(field, value);
        }
        return value;
      };
    // float, double: JSON value will be a number or one of the special string values "NaN", "Infinity", and "-Infinity".
    // Either numbers or strings are accepted. Exponent notation is also accepted.
    case ScalarType.FLOAT:
    case ScalarType.DOUBLE:
      return (opts, value) => {
        if (typeof value != "number") {
          throw errorScalarValue(field, value);
        }
        if (Number.isNaN(value))
          return "NaN";
        if (value === Number.POSITIVE_INFINITY)
          return "Infinity";
        if (value === Number.NEGATIVE_INFINITY)
          return "-Infinity";
        return value;
      };
    // string:
    case ScalarType.STRING:
      return (opts, value) => {
        if (typeof value != "string") {
          throw errorScalarValue(field, value);
        }
        return value;
      };
    // bool:
    case ScalarType.BOOL:
      return (opts, value) => {
        if (typeof value != "boolean") {
          throw errorScalarValue(field, value);
        }
        return value;
      };
    // JSON value will be a decimal string. Either numbers or strings are accepted.
    case ScalarType.UINT64:
    case ScalarType.FIXED64:
    case ScalarType.INT64:
    case ScalarType.SFIXED64:
    case ScalarType.SINT64:
      return (opts, value) => {
        if (typeof value == "bigint" || typeof value == "string" || typeof value == "number" && Number.isInteger(value)) {
          return value.toString();
        }
        throw errorScalarValue(field, value);
      };
    // bytes: JSON value will be the data encoded as a string using standard base64 encoding with paddings.
    // Either standard or URL-safe base64 encoding with/without paddings are accepted.
    case ScalarType.BYTES:
      return (opts, value) => {
        if (value instanceof Uint8Array) {
          return base64Encode(value);
        }
        throw errorScalarValue(field, value);
      };
  }
}
function errorScalarValue(field, value) {
  var _a;
  return new Error(`cannot encode ${field} to JSON: ${(_a = checkField(field, value)) === null || _a === void 0 ? void 0 : _a.message}`);
}
function writeExtensions(json, opts, registry, message, desc) {
  const unknown = message.$unknown;
  if (unknown === void 0) {
    return;
  }
  const tagSeen = /* @__PURE__ */ new Set();
  for (let i = 0; i < unknown.length; i++) {
    const { no } = unknown[i];
    if (!tagSeen.has(no)) {
      tagSeen.add(no);
      const extension = registry.getExtensionFor(desc, no);
      if (!extension) {
        continue;
      }
      const value = getExtension(message, extension);
      const [container, field] = createExtensionContainer(extension, value);
      const local = container[unsafeLocal];
      const jsonValue = compileFieldValue(field)(opts, local[field.localName]);
      if (jsonValue !== void 0) {
        json[extension.jsonName] = jsonValue;
      }
    }
  }
}
function anyToJson(val, opts) {
  if (val.typeUrl === "") {
    return {};
  }
  const { registry } = opts;
  let message;
  let desc;
  if (registry) {
    message = anyUnpack(val, registry);
    if (message) {
      desc = registry.getMessage(message.$typeName);
    }
  }
  if (!desc || !message) {
    throw new Error(`cannot encode message ${val.$typeName} to JSON: "${val.typeUrl}" is not in the type registry`);
  }
  const json = hasCustomJsonRepresentation(desc) ? {
    value: compiledWriter2(desc)(opts, message)
  } : compiledWriter2(desc)(opts, message);
  json["@type"] = val.typeUrl;
  return json;
}
function durationToJson(val) {
  const seconds = Number(val.seconds);
  const nanos = val.nanos;
  if (seconds > durationSecondsMax || seconds < durationSecondsMin) {
    throw new Error(`cannot encode message ${val.$typeName} to JSON: value out of range`);
  }
  if (seconds > 0 && nanos < 0 || seconds < 0 && nanos > 0) {
    throw new Error(`cannot encode message ${val.$typeName} to JSON: nanos sign must match seconds sign`);
  }
  let text = val.seconds.toString();
  if (nanos !== 0) {
    let nanosStr = Math.abs(nanos).toString();
    nanosStr = "0".repeat(9 - nanosStr.length) + nanosStr;
    if (nanosStr.substring(3) === "000000") {
      nanosStr = nanosStr.substring(0, 3);
    } else if (nanosStr.substring(6) === "000") {
      nanosStr = nanosStr.substring(0, 6);
    }
    text += "." + nanosStr;
    if (nanos < 0 && seconds == 0) {
      text = "-" + text;
    }
  }
  return text + "s";
}
function fieldMaskToJson(val) {
  return val.paths.map((p) => {
    if (protoSnakeCase(protoCamelCase(p)) !== p) {
      throw new Error(`cannot encode message ${val.$typeName} to JSON: lowerCamelCase of path name "${p}" is irreversible`);
    }
    return protoCamelCase(p);
  }).join(",");
}
function structToJson(val) {
  const json = {};
  const keys = Object.keys(val.fields);
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    setOwn(json, key, valueToJson(val.fields[key]));
  }
  return json;
}
function valueToJson(val) {
  switch (val.kind.case) {
    case "nullValue":
      return null;
    case "numberValue":
      if (!Number.isFinite(val.kind.value)) {
        throw new Error(`${val.$typeName} cannot be NaN or Infinity`);
      }
      return val.kind.value;
    case "boolValue":
      return val.kind.value;
    case "stringValue":
      return val.kind.value;
    case "structValue":
      return structToJson(val.kind.value);
    case "listValue":
      return listValueToJson(val.kind.value);
    default:
      throw new Error(`${val.$typeName} must have a value`);
  }
}
function listValueToJson(val) {
  return val.values.map(valueToJson);
}
function timestampToJson(val) {
  const ms = Number(val.seconds) * 1e3;
  if (ms < timestampMsMin || ms > timestampMsMax) {
    throw new Error(`cannot encode message ${val.$typeName} to JSON: must be from 0001-01-01T00:00:00Z to 9999-12-31T23:59:59Z inclusive`);
  }
  if (val.nanos < 0) {
    throw new Error(`cannot encode message ${val.$typeName} to JSON: nanos must not be negative`);
  }
  if (val.nanos > 999999999) {
    throw new Error(`cannot encode message ${val.$typeName} to JSON: nanos must not be greater than 99999999`);
  }
  let z = "Z";
  if (val.nanos > 0) {
    const nanosStr = (val.nanos + 1e9).toString().substring(1);
    if (nanosStr.substring(3) === "000000") {
      z = "." + nanosStr.substring(0, 3) + "Z";
    } else if (nanosStr.substring(6) === "000") {
      z = "." + nanosStr.substring(0, 6) + "Z";
    } else {
      z = "." + nanosStr + "Z";
    }
  }
  return new Date(ms).toISOString().replace(".000Z", z);
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/from-json.js
var IMPLICIT6 = 2;
function makeReadContext2(options2) {
  return Object.assign(Object.assign({ ignoreUnknownFields: false, recursionLimit: 100 }, options2), { depth: 0 });
}
function fromJsonString(schema, json, options2) {
  return fromJson(schema, parseJsonString(json, schema.typeName), options2);
}
function fromJson(schema, json, options2) {
  const message = create(schema);
  readMessage(schema, message, json, options2);
  return message;
}
function readMessage(schema, message, json, options2) {
  try {
    compiledReader2(schema)(message, json, makeReadContext2(options2));
  } catch (e) {
    if (isFieldError(e)) {
      throw new Error(`cannot decode ${e.field()} from JSON: ${e.message}`, {
        cause: e
      });
    }
    throw e;
  }
}
var compiledReaders2 = /* @__PURE__ */ new WeakMap();
function compiledReader2(desc) {
  let compiled = compiledReaders2.get(desc);
  if (compiled === void 0) {
    compiled = compileMessage4(desc);
  }
  return compiled;
}
function compileMessage4(desc) {
  const descString = String(desc);
  const readWkt = compileWkt2(desc);
  if (readWkt !== void 0) {
    const compiled2 = (message, json, ctx) => {
      if (++ctx.depth > ctx.recursionLimit) {
        throw new Error(`cannot decode ${descString} from JSON: maximum recursion depth of ${ctx.recursionLimit} reached`);
      }
      readWkt(message, json, ctx);
      ctx.depth--;
    };
    compiledReaders2.set(desc, compiled2);
    return compiled2;
  }
  const typeName = desc.typeName;
  const fieldsByJsonKey = /* @__PURE__ */ new Map();
  const compiled = (message, json, ctx) => {
    var _a;
    if (++ctx.depth > ctx.recursionLimit) {
      throw new Error(`cannot decode ${descString} from JSON: maximum recursion depth of ${ctx.recursionLimit} reached`);
    }
    if (json == null || Array.isArray(json) || typeof json != "object") {
      throw new Error(`cannot decode ${descString} from JSON: ${formatVal(json)}`);
    }
    const oneofSeen = /* @__PURE__ */ new Map();
    const fieldSeen = /* @__PURE__ */ new Set();
    const jsonKeys = Object.keys(json);
    for (let i = 0; i < jsonKeys.length; i++) {
      const jsonKey = jsonKeys[i];
      const jsonValue = json[jsonKey];
      const entry = fieldsByJsonKey.get(jsonKey);
      if (entry !== void 0) {
        const field = entry.field;
        if (fieldSeen.has(field)) {
          throw new FieldError(field, "set multiple times");
        }
        fieldSeen.add(field);
        if (entry.oneofScalarNullSkip && jsonValue === null) {
          continue;
        }
        if (entry.oneof) {
          const seen = oneofSeen.get(entry.oneof);
          if (seen !== void 0) {
            throw new FieldError(entry.oneof, `oneof set multiple times by ${seen.name} and ${field.name}`);
          }
          oneofSeen.set(entry.oneof, field);
        }
        entry.read(message, jsonValue, ctx);
      } else {
        const extension = jsonKey.startsWith("[") && jsonKey.endsWith("]") ? (_a = ctx.registry) === null || _a === void 0 ? void 0 : _a.getExtension(jsonKey.substring(1, jsonKey.length - 1)) : void 0;
        if ((extension === null || extension === void 0 ? void 0 : extension.extendee.typeName) == typeName) {
          const [container, field, get] = createExtensionContainer(extension);
          compileFieldReader2(field)(container[unsafeLocal], jsonValue, ctx);
          setExtension(message, extension, get());
        }
        if (extension === void 0 && !ctx.ignoreUnknownFields) {
          throw new Error(`cannot decode ${descString} from JSON: key "${jsonKey}" is unknown`);
        }
      }
    }
    ctx.depth--;
  };
  compiledReaders2.set(desc, compiled);
  for (const field of desc.fields) {
    const entry = {
      read: compileFieldReader2(field),
      field,
      oneof: field.oneof,
      oneofScalarNullSkip: field.oneof !== void 0 && field.fieldKind == "scalar"
    };
    fieldsByJsonKey.set(field.name, entry).set(field.jsonName, entry);
  }
  return compiled;
}
function compileWkt2(desc) {
  if (!desc.typeName.startsWith("google.protobuf.")) {
    return void 0;
  }
  switch (desc.typeName) {
    case "google.protobuf.Any":
      return (message, json, ctx) => anyFromJson(message, json, ctx);
    case "google.protobuf.Timestamp":
      return (message, json) => timestampFromJson(message, json);
    case "google.protobuf.Duration":
      return (message, json) => durationFromJson(message, json);
    case "google.protobuf.FieldMask":
      return (message, json) => fieldMaskFromJson(message, json);
    case "google.protobuf.Struct":
      return (message, json, ctx) => structFromJson(message, json, ctx);
    case "google.protobuf.Value":
      return (message, json, ctx) => valueFromJson(message, json, ctx);
    case "google.protobuf.ListValue":
      return (message, json, ctx) => listValueFromJson(message, json, ctx);
    default:
      if (isWrapperDesc(desc)) {
        const valueField = desc.fields[0];
        const localName = valueField.localName;
        const scalar = valueField.scalar;
        const longAsString = valueField.longAsString;
        const readScalar = compileScalarConverter(valueField);
        return (message, json) => {
          if (json === null) {
            message[localName] = scalarZeroValue(scalar, longAsString);
          } else {
            message[localName] = readScalar(json);
          }
        };
      }
      return void 0;
  }
}
function compileFieldReader2(field) {
  switch (field.fieldKind) {
    case "scalar":
      return compileScalarFieldReader2(field);
    case "enum":
      return compileEnumFieldReader2(field);
    case "message":
      return compileMessageFieldReader2(field);
    case "list":
      return compileListFieldReader2(field);
    case "map":
      return compileMapFieldReader2(field);
  }
}
function compileScalarFieldReader2(field) {
  const readScalar = compileScalarConverter(field);
  const localName = field.localName;
  if (field.oneof) {
    const oneofLocalName = field.oneof.localName;
    return (message, json) => {
      message[oneofLocalName] = {
        case: localName,
        value: readScalar(json)
      };
    };
  }
  const clear = compileClear(field);
  return (message, json) => {
    if (json === null) {
      clear(message);
    } else {
      message[localName] = readScalar(json);
    }
  };
}
function compileClear(field) {
  const localName = field.localName;
  if (field.presence != IMPLICIT6) {
    return (message) => {
      delete message[localName];
    };
  }
  if (field.fieldKind == "enum") {
    const zero = field.enum.values[0].number;
    return (message) => {
      message[localName] = zero;
    };
  }
  const scalar = field.scalar;
  const longAsString = field.longAsString;
  return (message) => {
    message[localName] = scalarZeroValue(scalar, longAsString);
  };
}
function compileEnumFieldReader2(field) {
  const readEnumValue = compileEnumConverter(field.enum);
  const checkEnum = compileEnumCheck(field.enum);
  const localName = field.localName;
  const nullResets = field.enum.typeName != "google.protobuf.NullValue";
  if (field.oneof) {
    const oneofLocalName = field.oneof.localName;
    return (message, json, ctx) => {
      if (json === null && nullResets) {
        const oneof = message[oneofLocalName];
        if (oneof.case === localName) {
          message[oneofLocalName] = { case: void 0 };
        }
        return;
      }
      const value = readEnumValue(json, ctx.ignoreUnknownFields);
      if (value === tokenIgnoredUnknownEnum) {
        return;
      }
      const check = checkEnum(value);
      if (check !== true) {
        throw new FieldError(field, reasonSingular(field, value, check));
      }
      message[oneofLocalName] = { case: localName, value };
    };
  }
  const clear = compileClear(field);
  return (message, json, ctx) => {
    if (json === null && nullResets) {
      clear(message);
      return;
    }
    const value = readEnumValue(json, ctx.ignoreUnknownFields);
    if (value === tokenIgnoredUnknownEnum) {
      return;
    }
    const check = checkEnum(value);
    if (check !== true) {
      throw new FieldError(field, reasonSingular(field, value, check));
    }
    message[localName] = value;
  };
}
function compileMessageFieldReader2(field) {
  const localName = field.localName;
  const { toMessage, toLocal } = localMessageMapper(field);
  const readChild = compiledReader2(field.message);
  const nullResets = field.message.typeName != "google.protobuf.Value";
  if (field.oneof) {
    const oneofLocalName = field.oneof.localName;
    return (message, json, ctx) => {
      const oneof = message[oneofLocalName];
      if (json === null && nullResets) {
        if (oneof.case === localName) {
          message[oneofLocalName] = { case: void 0 };
        }
        return;
      }
      const child = toMessage(oneof.case === localName ? oneof.value : void 0);
      readChild(child, json, ctx);
      message[oneofLocalName] = { case: localName, value: toLocal(child) };
    };
  }
  return (message, json, ctx) => {
    if (json === null && nullResets) {
      delete message[localName];
      return;
    }
    const child = toMessage(message[localName]);
    readChild(child, json, ctx);
    message[localName] = toLocal(child);
  };
}
function compileListFieldReader2(field) {
  const localName = field.localName;
  const readItem = compileListItemReader(field);
  return (message, json, ctx) => {
    if (json === null) {
      return;
    }
    if (!Array.isArray(json)) {
      throw new FieldError(field, "expected Array, got " + formatVal(json));
    }
    const items = message[localName];
    for (let i = 0; i < json.length; i++) {
      const value = readItem(json[i], ctx, items.length);
      if (value !== tokenIgnoredUnknownEnum) {
        items.push(value);
      }
    }
  };
}
function compileListItemReader(field) {
  switch (field.listKind) {
    case "scalar": {
      const parseScalar = compileScalarParse(field);
      const checkValue = checkScalarValue(field.scalar);
      const toLocal = compileScalarToLocal(field);
      return (json, ctx, index) => {
        if (json === null) {
          throw new FieldError(field, "list item must not be null");
        }
        const value = parseScalar(json);
        const check = checkValue(value);
        if (check !== true) {
          throw new FieldError(field, `list item #${index + 1}: ${reasonSingular(field, value, check)}`);
        }
        return toLocal(value);
      };
    }
    case "enum": {
      const readEnumValue = compileEnumConverter(field.enum);
      const checkEnum = compileEnumCheck(field.enum);
      const nullResets = field.enum.typeName != "google.protobuf.NullValue";
      return (json, ctx, index) => {
        if (json === null && nullResets) {
          throw new FieldError(field, "list item must not be null");
        }
        const value = readEnumValue(json, ctx.ignoreUnknownFields);
        if (value === tokenIgnoredUnknownEnum) {
          return value;
        }
        const check = checkEnum(value);
        if (check !== true) {
          throw new FieldError(field, `list item #${index + 1}: ${reasonSingular(field, value, check)}`);
        }
        return value;
      };
    }
    case "message": {
      const { toMessage, toLocal } = localMessageMapper(field);
      const readChild = compiledReader2(field.message);
      const nullResets = field.message.typeName != "google.protobuf.Value";
      return (json, ctx) => {
        if (json === null && nullResets) {
          throw new FieldError(field, "list item must not be null");
        }
        const child = toMessage(void 0);
        readChild(child, json, ctx);
        return toLocal(child);
      };
    }
  }
}
function compileMapFieldReader2(field) {
  const localName = field.localName;
  const mapKey = field.mapKey;
  const parseMapKey = compileMapKeyParse(mapKey);
  const checkMapKey = checkScalarValue(mapKey);
  let parseValue;
  let checkValue;
  let toLocalValue = (value) => value;
  let nullResets = true;
  switch (field.mapKind) {
    case "scalar": {
      parseValue = compileScalarParse(field);
      checkValue = checkScalarValue(field.scalar);
      toLocalValue = compileScalarToLocal(field);
      break;
    }
    case "enum": {
      const readEnumValue = compileEnumConverter(field.enum);
      parseValue = (json, ctx) => readEnumValue(json, ctx.ignoreUnknownFields);
      checkValue = compileEnumCheck(field.enum);
      nullResets = field.enum.typeName != "google.protobuf.NullValue";
      break;
    }
    case "message": {
      const { toMessage, toLocal } = localMessageMapper(field);
      const readChild = compiledReader2(field.message);
      nullResets = field.message.typeName != "google.protobuf.Value";
      parseValue = (json, ctx) => {
        const child = toMessage(void 0);
        readChild(child, json, ctx);
        return toLocal(child);
      };
      break;
    }
  }
  return (message, json, ctx) => {
    if (json === null) {
      return;
    }
    if (typeof json != "object" || Array.isArray(json)) {
      throw new FieldError(field, "expected object, got " + formatVal(json));
    }
    const record = message[localName];
    const seen = /* @__PURE__ */ new Set();
    const jsonMapKeys = Object.keys(json);
    for (let i = 0; i < jsonMapKeys.length; i++) {
      const jsonMapKey = jsonMapKeys[i];
      const jsonMapValue = json[jsonMapKey];
      const key = parseMapKey(jsonMapKey);
      if (seen.has(key)) {
        throw new FieldError(field, `duplicate map key "${jsonMapKey}"`);
      }
      seen.add(key);
      if (jsonMapValue === null && nullResets) {
        throw new FieldError(field, "map value must not be null");
      }
      const value = parseValue(jsonMapValue, ctx);
      if (value === tokenIgnoredUnknownEnum) {
        continue;
      }
      const checkKey = checkMapKey(key);
      if (checkKey !== true) {
        throw new FieldError(field, `invalid map key: ${reasonSingular({ scalar: mapKey }, key, checkKey)}`);
      }
      if (checkValue !== void 0) {
        const check = checkValue(value);
        if (check !== true) {
          throw new FieldError(field, `map entry ${formatVal(key)}: ${reasonSingular(field, value, check)}`);
        }
      }
      setOwn(record, key, toLocalValue(value));
    }
  };
}
var tokenIgnoredUnknownEnum = /* @__PURE__ */ Symbol();
function compileEnumConverter(desc) {
  const zero = desc.values[0].number;
  const numbers = /* @__PURE__ */ new Map();
  for (const value of desc.values) {
    numbers.set(value.jsonName, value.number);
  }
  for (const value of desc.values) {
    numbers.set(value.name, value.number);
  }
  return (json, ignoreUnknownFields) => {
    if (json === null) {
      return zero;
    }
    switch (typeof json) {
      case "number":
        if (Number.isInteger(json)) {
          return json;
        }
        break;
      case "string": {
        const number = numbers.get(json);
        if (number !== void 0) {
          return number;
        }
        if (ignoreUnknownFields) {
          return tokenIgnoredUnknownEnum;
        }
        break;
      }
    }
    throw new Error(`cannot decode ${desc} from JSON: ${formatVal(json)}`);
  };
}
function compileEnumCheck(desc) {
  if (desc.open) {
    return checkScalarValue(ScalarType.INT32);
  }
  const values = desc.values;
  return (value) => values.some((v) => v.number === value);
}
function compileScalarConverter(field) {
  const parseScalar = compileScalarParse(field);
  const checkValue = checkScalarValue(field.scalar);
  const toLocal = compileScalarToLocal(field);
  return (json) => {
    const value = parseScalar(json);
    const check = checkValue(value);
    if (check !== true) {
      throw new FieldError(field, reasonSingular(field, value, check));
    }
    return toLocal(value);
  };
}
function compileScalarParse(field) {
  switch (field.scalar) {
    // float, double: JSON value will be a number or one of the special string values "NaN", "Infinity", and "-Infinity".
    // Either numbers or strings are accepted. Exponent notation is also accepted.
    case ScalarType.DOUBLE:
    case ScalarType.FLOAT:
      return (json) => {
        if (json === "NaN")
          return NaN;
        if (json === "Infinity")
          return Number.POSITIVE_INFINITY;
        if (json === "-Infinity")
          return Number.NEGATIVE_INFINITY;
        if (typeof json == "number") {
          if (Number.isNaN(json)) {
            throw new FieldError(field, "unexpected NaN number");
          }
          if (!Number.isFinite(json)) {
            throw new FieldError(field, "unexpected infinite number");
          }
          return json;
        }
        if (typeof json == "string") {
          if (json === "") {
            return json;
          }
          if (json.trim().length !== json.length) {
            return json;
          }
          const float = Number(json);
          if (!Number.isFinite(float)) {
            return json;
          }
          return float;
        }
        return json;
      };
    // int32, fixed32, uint32: JSON value will be a decimal number. Either numbers or strings are accepted.
    case ScalarType.INT32:
    case ScalarType.FIXED32:
    case ScalarType.SFIXED32:
    case ScalarType.SINT32:
    case ScalarType.UINT32:
      return int32FromJson;
    // bytes: JSON value will be the data encoded as a string using standard base64 encoding with paddings.
    // Either standard or URL-safe base64 encoding with/without paddings are accepted.
    case ScalarType.BYTES:
      return (json) => {
        if (typeof json == "string") {
          if (json === "") {
            return new Uint8Array(0);
          }
          try {
            return base64Decode(json);
          } catch (e) {
            const message = e instanceof Error ? e.message : String(e);
            throw new FieldError(field, message);
          }
        }
        return json;
      };
    // int64, sfixed64, sint64, fixed64, uint64: The validation step accepts
    // string and number. string, bool: no conversion.
    default:
      return (json) => json;
  }
}
function compileScalarToLocal(field) {
  const longAsString = field.fieldKind !== "map" && field.longAsString;
  switch (field.scalar) {
    case ScalarType.INT64:
    case ScalarType.SFIXED64:
    case ScalarType.SINT64:
      if (longAsString) {
        return (value) => String(value);
      }
      return (value) => typeof value == "string" || typeof value == "number" ? protoInt64.parse(value) : value;
    case ScalarType.FIXED64:
    case ScalarType.UINT64:
      if (longAsString) {
        return (value) => String(value);
      }
      return (value) => typeof value == "string" || typeof value == "number" ? protoInt64.uParse(value) : value;
    default:
      return (value) => value;
  }
}
function compileMapKeyParse(type) {
  switch (type) {
    case ScalarType.BOOL:
      return (jsonString) => {
        switch (jsonString) {
          case "true":
            return true;
          case "false":
            return false;
        }
        return jsonString;
      };
    case ScalarType.INT32:
    case ScalarType.FIXED32:
    case ScalarType.UINT32:
    case ScalarType.SFIXED32:
    case ScalarType.SINT32:
      return int32FromJson;
    case ScalarType.INT64:
    case ScalarType.SINT64:
    case ScalarType.SFIXED64:
    case ScalarType.UINT64:
    case ScalarType.FIXED64:
      return (jsonString) => /^-?0+$/.test(jsonString) ? "0" : jsonString.replace(/^(-?)0+(?=\d)/, "$1");
    default:
      return (jsonString) => jsonString;
  }
}
function int32FromJson(json) {
  if (typeof json == "string") {
    if (json === "") {
      return json;
    }
    if (json.trim().length !== json.length) {
      return json;
    }
    const num = Number(json);
    if (Number.isNaN(num)) {
      return json;
    }
    return num;
  }
  return json;
}
function parseJsonString(jsonString, typeName) {
  let json;
  try {
    json = JSON.parse(jsonString);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    throw new Error(
      `cannot decode message ${typeName} from JSON: ${message}`,
      // @ts-expect-error we use the ES2022 error CTOR option "cause" for better stack traces
      { cause: e }
    );
  }
  checkDuplicateKeys(jsonString, typeName);
  return json;
}
function checkDuplicateKeys(jsonString, typeName) {
  const stack = [];
  let expectKey = false;
  let i = 0;
  while (i < jsonString.length) {
    switch (jsonString[i]) {
      case "{":
        stack.push(/* @__PURE__ */ new Set());
        expectKey = true;
        i++;
        break;
      case "[":
        stack.push(null);
        expectKey = false;
        i++;
        break;
      case "}":
      case "]":
        stack.pop();
        expectKey = false;
        i++;
        break;
      case ",":
        expectKey = stack[stack.length - 1] != null;
        i++;
        break;
      case ":":
        expectKey = false;
        i++;
        break;
      case '"': {
        const open = i++;
        let escaped = false;
        while (i < jsonString.length) {
          if (jsonString[i] == "\\") {
            escaped = true;
            i += 2;
            continue;
          }
          if (jsonString[i] == '"') {
            break;
          }
          i++;
        }
        const close = i++;
        const seen = stack[stack.length - 1];
        if (expectKey && seen) {
          const name = escaped ? JSON.parse(jsonString.substring(open, close + 1)) : jsonString.substring(open + 1, close);
          if (seen.has(name)) {
            throw new Error(`cannot decode message ${typeName} from JSON: duplicate object key "${name}"`);
          }
          seen.add(name);
        }
        expectKey = false;
        break;
      }
      default:
        i++;
        break;
    }
  }
}
function anyFromJson(any, json, ctx) {
  var _a;
  if (json === null || Array.isArray(json) || typeof json != "object") {
    throw new Error(`cannot decode message ${any.$typeName} from JSON: expected object but got ${formatVal(json)}`);
  }
  if (Object.keys(json).length == 0) {
    return;
  }
  const typeUrl = json["@type"];
  if (typeof typeUrl != "string" || typeUrl == "") {
    throw new Error(`cannot decode message ${any.$typeName} from JSON: "@type" is empty`);
  }
  const typeName = typeUrl.includes("/") ? typeUrl.substring(typeUrl.lastIndexOf("/") + 1) : typeUrl;
  if (!typeName.length) {
    throw new Error(`cannot decode message ${any.$typeName} from JSON: "@type" is invalid`);
  }
  const desc = (_a = ctx.registry) === null || _a === void 0 ? void 0 : _a.getMessage(typeName);
  if (!desc) {
    throw new Error(`cannot decode message ${any.$typeName} from JSON: ${typeUrl} is not in the type registry`);
  }
  const message = create(desc);
  if (hasCustomJsonRepresentation(desc) && Object.prototype.hasOwnProperty.call(json, "value")) {
    compiledReader2(desc)(message, json.value, ctx);
  } else {
    const copy = {};
    for (const key of Object.keys(json)) {
      if (key !== "@type") {
        setOwn(copy, key, json[key]);
      }
    }
    compiledReader2(desc)(message, copy, ctx);
  }
  anyPack(desc, message, any);
}
function timestampFromJson(timestamp, json) {
  if (typeof json !== "string") {
    throw new Error(`cannot decode message ${timestamp.$typeName} from JSON: ${formatVal(json)}`);
  }
  const matches = json.match(/^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,9}))?(?:Z|([+-][0-9][0-9]:[0-9][0-9]))$/);
  if (!matches) {
    throw new Error(`cannot decode message ${timestamp.$typeName} from JSON: invalid RFC 3339 string`);
  }
  const ms = Date.parse(
    // biome-ignore format: want this to read well
    matches[1] + "-" + matches[2] + "-" + matches[3] + "T" + matches[4] + ":" + matches[5] + ":" + matches[6] + (matches[8] ? matches[8] : "Z")
  );
  if (Number.isNaN(ms)) {
    throw new Error(`cannot decode message ${timestamp.$typeName} from JSON: invalid RFC 3339 string`);
  }
  if (ms < timestampMsMin || ms > timestampMsMax) {
    throw new Error(`cannot decode message ${timestamp.$typeName} from JSON: must be from 0001-01-01T00:00:00Z to 9999-12-31T23:59:59Z inclusive`);
  }
  timestamp.seconds = protoInt64.parse(ms / 1e3);
  timestamp.nanos = 0;
  if (matches[7]) {
    timestamp.nanos = parseInt("1" + matches[7] + "0".repeat(9 - matches[7].length)) - 1e9;
  }
}
function durationFromJson(duration, json) {
  if (typeof json !== "string") {
    throw new Error(`cannot decode message ${duration.$typeName} from JSON: ${formatVal(json)}`);
  }
  const match = json.match(/^(-?[0-9]+)(?:\.([0-9]+))?s/);
  if (match === null) {
    throw new Error(`cannot decode message ${duration.$typeName} from JSON: ${formatVal(json)}`);
  }
  const longSeconds = Number(match[1]);
  if (longSeconds > durationSecondsMax || longSeconds < durationSecondsMin) {
    throw new Error(`cannot decode message ${duration.$typeName} from JSON: ${formatVal(json)}`);
  }
  duration.seconds = protoInt64.parse(longSeconds);
  if (typeof match[2] !== "string") {
    return;
  }
  const nanosStr = match[2] + "0".repeat(9 - match[2].length);
  duration.nanos = parseInt(nanosStr);
  if (longSeconds < 0 || Object.is(longSeconds, -0)) {
    duration.nanos = -duration.nanos;
  }
}
function fieldMaskFromJson(fieldMask, json) {
  if (typeof json !== "string") {
    throw new Error(`cannot decode message ${fieldMask.$typeName} from JSON: ${formatVal(json)}`);
  }
  if (json === "") {
    return;
  }
  fieldMask.paths = json.split(",").map((path) => {
    if (path.includes("_")) {
      throw new Error(`cannot decode message ${fieldMask.$typeName} from JSON: path names must be lowerCamelCase`);
    }
    return protoSnakeCase(path);
  });
}
function structFromJson(struct, json, ctx) {
  if (typeof json != "object" || json == null || Array.isArray(json)) {
    throw new Error(`cannot decode message ${struct.$typeName} from JSON ${formatVal(json)}`);
  }
  const keys = Object.keys(json);
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    const parsedValue = create(ValueSchema);
    valueFromJson(parsedValue, json[key], ctx);
    setOwn(struct.fields, key, parsedValue);
  }
}
function valueFromJson(value, json, ctx) {
  if (++ctx.depth > ctx.recursionLimit) {
    throw new Error(`cannot decode ${value.$typeName} from JSON: maximum recursion depth of ${ctx.recursionLimit} reached`);
  }
  switch (typeof json) {
    case "number":
      value.kind = { case: "numberValue", value: json };
      break;
    case "string":
      value.kind = { case: "stringValue", value: json };
      break;
    case "boolean":
      value.kind = { case: "boolValue", value: json };
      break;
    case "object":
      if (json === null) {
        value.kind = { case: "nullValue", value: NullValue.NULL_VALUE };
      } else if (Array.isArray(json)) {
        const listValue = create(ListValueSchema);
        listValueFromJson(listValue, json, ctx);
        value.kind = { case: "listValue", value: listValue };
      } else {
        const struct = create(StructSchema);
        structFromJson(struct, json, ctx);
        value.kind = { case: "structValue", value: struct };
      }
      break;
    default:
      throw new Error(`cannot decode message ${value.$typeName} from JSON ${formatVal(json)}`);
  }
  ctx.depth--;
  return value;
}
function listValueFromJson(listValue, json, ctx) {
  if (!Array.isArray(json)) {
    throw new Error(`cannot decode message ${listValue.$typeName} from JSON ${formatVal(json)}`);
  }
  for (let i = 0; i < json.length; i++) {
    const value = create(ValueSchema);
    valueFromJson(value, json[i], ctx);
    listValue.values.push(value);
  }
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/code-string.js
function codeToString(value) {
  const name = Code[value];
  if (typeof name != "string") {
    return value.toString();
  }
  return name[0].toLowerCase() + name.substring(1).replace(/[A-Z]/g, (c) => "_" + c.toLowerCase());
}
var stringToCode;
function codeFromString(value) {
  if (!stringToCode) {
    stringToCode = {};
    for (const value2 of Object.values(Code)) {
      if (typeof value2 == "string") {
        continue;
      }
      stringToCode[codeToString(value2)] = value2;
    }
  }
  return stringToCode[value];
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/connect-error.js
var ConnectError = class _ConnectError extends Error {
  /**
   * Create a new ConnectError.
   * If no code is provided, code "unknown" is used.
   * Outgoing details are only relevant for the server side - a service may
   * raise an error with details, and it is up to the protocol implementation
   * to encode and send the details along with the error.
   */
  constructor(message, code = Code.Unknown, metadata, outgoingDetails, cause) {
    super(createMessage(message, code));
    this.name = "ConnectError";
    this.isWireError = false;
    Object.setPrototypeOf(this, new.target.prototype);
    this.rawMessage = message;
    this.code = code;
    this.metadata = new Headers(metadata !== null && metadata !== void 0 ? metadata : {});
    this.details = outgoingDetails !== null && outgoingDetails !== void 0 ? outgoingDetails : [];
    this.cause = cause;
  }
  /**
   * Convert any value - typically a caught error into a ConnectError,
   * following these rules:
   * - If the value is already a ConnectError, return it as is.
   * - If the value is an AbortError or TimeoutError from the fetch API, return
   *   the message of the error with code Canceled.
   * - For other Errors, return the error message with code Unknown by default.
   * - For other values, return the values String representation as a message,
   *   with the code Unknown by default.
   * The original value will be used for the "cause" property for the new
   * ConnectError.
   */
  static from(reason, code = Code.Unknown) {
    if (reason instanceof _ConnectError) {
      return reason;
    }
    if (reason instanceof Error) {
      if (reason.name == "AbortError" || reason.name == "TimeoutError") {
        return new _ConnectError(reason.message, Code.Canceled);
      }
      return new _ConnectError(reason.message, code, void 0, void 0, reason);
    }
    return new _ConnectError(String(reason), code, void 0, void 0, reason);
  }
  static [Symbol.hasInstance](v) {
    if (!(v instanceof Error)) {
      return false;
    }
    if (Object.getPrototypeOf(v) === _ConnectError.prototype) {
      return true;
    }
    return v.name === "ConnectError" && "code" in v && typeof v.code === "number" && "metadata" in v && "details" in v && Array.isArray(v.details) && "rawMessage" in v && typeof v.rawMessage == "string" && "cause" in v;
  }
  findDetails(typeOrRegistry) {
    const registry = typeOrRegistry.kind === "message" ? {
      getMessage: (typeName) => typeName === typeOrRegistry.typeName ? typeOrRegistry : void 0
    } : typeOrRegistry;
    const details = [];
    for (const data of this.details) {
      if ("desc" in data) {
        if (registry.getMessage(data.desc.typeName)) {
          details.push(create(data.desc, data.value));
        }
        continue;
      }
      const desc = registry.getMessage(data.type);
      if (desc) {
        try {
          details.push(fromBinary(desc, data.value));
        } catch (_) {
        }
      }
    }
    return details;
  }
};
function createMessage(message, code) {
  return message.length ? `[${codeToString(code)}] ${message}` : `[${codeToString(code)}]`;
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/http-headers.js
function appendHeaders(...headers) {
  const h = new Headers();
  for (const e of headers) {
    e.forEach((value, key) => {
      h.append(key, value);
    });
  }
  return h;
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/any-client.js
function makeAnyClient(service, createMethod) {
  const client2 = {};
  for (const desc of service.methods) {
    const method = createMethod(desc);
    if (method != null) {
      client2[desc.localName] = method;
    }
  }
  return client2;
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol/compression.js
var compressedFlag = 1;

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol/limit-io.js
function assertReadMaxBytes(readMaxBytes, bytesRead, totalSizeKnown = false) {
  if (bytesRead > readMaxBytes) {
    let message = `message size is larger than configured readMaxBytes ${readMaxBytes}`;
    if (totalSizeKnown) {
      message = `message size ${bytesRead} is larger than configured readMaxBytes ${readMaxBytes}`;
    }
    throw new ConnectError(message, Code.ResourceExhausted);
  }
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol/envelope.js
function createEnvelopeDecoder(readMaxBytes) {
  return new EnvelopeDecoderImpl(readMaxBytes);
}
var EnvelopeDecoderImpl = class {
  constructor(readMaxBytes) {
    this.readMaxBytes = readMaxBytes;
    this.header = new Uint8Array(5);
    this.headerView = new DataView(this.header.buffer);
    this.buf = [];
  }
  get byteLength() {
    return this.buf.reduce((a, b) => a + b.byteLength, 0);
  }
  decode(chunk) {
    this.buf.push(chunk);
    const envs = [];
    for (; ; ) {
      let env = this.pop();
      if (!env) {
        break;
      }
      envs.push(env);
    }
    return envs;
  }
  // consume an enveloped message
  pop() {
    if (!this.env) {
      this.env = this.head();
      if (!this.env) {
        return void 0;
      }
    }
    if (this.cons(this.env.data)) {
      const env = this.env;
      this.env = void 0;
      return env;
    }
    return void 0;
  }
  // consume header
  head() {
    if (!this.cons(this.header)) {
      return void 0;
    }
    const flags = this.headerView.getUint8(0);
    const length = this.headerView.getUint32(1);
    assertReadMaxBytes(this.readMaxBytes, length, true);
    return {
      flags,
      data: new Uint8Array(length)
    };
  }
  // consume from buffer, fill target
  cons(target) {
    const wantLength = target.byteLength;
    if (this.byteLength < wantLength) {
      return false;
    }
    let offset = 0;
    while (offset < wantLength) {
      const chunk = this.buf.shift();
      if (chunk.byteLength > wantLength - offset) {
        target.set(chunk.subarray(0, wantLength - offset), offset);
        this.buf.unshift(chunk.subarray(wantLength - offset));
        offset += wantLength - offset;
      } else {
        target.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }
    return true;
  }
};
function createEnvelopeReadableStream(stream) {
  let reader;
  const buffer = createEnvelopeDecoder(4294967295);
  return new ReadableStream({
    start() {
      reader = stream.getReader();
    },
    async pull(controller) {
      let enqueuedOnce = false;
      while (!enqueuedOnce) {
        const result = await reader.read();
        if (result.done) {
          if (buffer.byteLength > 0) {
            controller.error(new ConnectError("protocol error: incomplete envelope", Code.InvalidArgument));
          }
          controller.close();
        } else {
          for (const env of buffer.decode(result.value)) {
            controller.enqueue(env);
            enqueuedOnce = true;
          }
        }
      }
    }
  });
}
function encodeEnvelope(flags, data) {
  const bytes2 = new Uint8Array(data.length + 5);
  bytes2.set(data, 5);
  const v = new DataView(bytes2.buffer, bytes2.byteOffset, bytes2.byteLength);
  v.setUint8(0, flags);
  v.setUint32(1, data.length);
  return bytes2;
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol/async-iterable.js
var __asyncValues = function(o) {
  if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
  var m = o[Symbol.asyncIterator], i;
  return m ? m.call(o) : (o = typeof __values === "function" ? __values(o) : o[Symbol.iterator](), i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function() {
    return this;
  }, i);
  function verb(n) {
    i[n] = o[n] && function(v) {
      return new Promise(function(resolve, reject) {
        v = o[n](v), settle(resolve, reject, v.done, v.value);
      });
    };
  }
  function settle(resolve, reject, d, v) {
    Promise.resolve(v).then(function(v2) {
      resolve({ value: v2, done: d });
    }, reject);
  }
};
var __await = function(v) {
  return this instanceof __await ? (this.v = v, this) : new __await(v);
};
var __asyncGenerator = function(thisArg, _arguments, generator) {
  if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
  var g = generator.apply(thisArg, _arguments || []), i, q = [];
  return i = Object.create((typeof AsyncIterator === "function" ? AsyncIterator : Object).prototype), verb("next"), verb("throw"), verb("return", awaitReturn), i[Symbol.asyncIterator] = function() {
    return this;
  }, i;
  function awaitReturn(f) {
    return function(v) {
      return Promise.resolve(v).then(f, reject);
    };
  }
  function verb(n, f) {
    if (g[n]) {
      i[n] = function(v) {
        return new Promise(function(a, b) {
          q.push([n, v, a, b]) > 1 || resume(n, v);
        });
      };
      if (f) i[n] = f(i[n]);
    }
  }
  function resume(n, v) {
    try {
      step(g[n](v));
    } catch (e) {
      settle(q[0][3], e);
    }
  }
  function step(r) {
    r.value instanceof __await ? Promise.resolve(r.value.v).then(fulfill, reject) : settle(q[0][2], r);
  }
  function fulfill(value) {
    resume("next", value);
  }
  function reject(value) {
    resume("throw", value);
  }
  function settle(f, v) {
    if (f(v), q.shift(), q.length) resume(q[0][0], q[0][1]);
  }
};
var __asyncDelegator = function(o) {
  var i, p;
  return i = {}, verb("next"), verb("throw", function(e) {
    throw e;
  }), verb("return"), i[Symbol.iterator] = function() {
    return this;
  }, i;
  function verb(n, f) {
    i[n] = o[n] ? function(v) {
      return (p = !p) ? { value: __await(o[n](v)), done: false } : f ? f(v) : v;
    } : f;
  }
};
function createAsyncIterable(items) {
  return __asyncGenerator(this, arguments, function* createAsyncIterable_1() {
    yield __await(yield* __asyncDelegator(__asyncValues(items)));
  });
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/promise-client.js
var __asyncValues2 = function(o) {
  if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
  var m = o[Symbol.asyncIterator], i;
  return m ? m.call(o) : (o = typeof __values === "function" ? __values(o) : o[Symbol.iterator](), i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function() {
    return this;
  }, i);
  function verb(n) {
    i[n] = o[n] && function(v) {
      return new Promise(function(resolve, reject) {
        v = o[n](v), settle(resolve, reject, v.done, v.value);
      });
    };
  }
  function settle(resolve, reject, d, v) {
    Promise.resolve(v).then(function(v2) {
      resolve({ value: v2, done: d });
    }, reject);
  }
};
var __await2 = function(v) {
  return this instanceof __await2 ? (this.v = v, this) : new __await2(v);
};
var __asyncDelegator2 = function(o) {
  var i, p;
  return i = {}, verb("next"), verb("throw", function(e) {
    throw e;
  }), verb("return"), i[Symbol.iterator] = function() {
    return this;
  }, i;
  function verb(n, f) {
    i[n] = o[n] ? function(v) {
      return (p = !p) ? { value: __await2(o[n](v)), done: false } : f ? f(v) : v;
    } : f;
  }
};
var __asyncGenerator2 = function(thisArg, _arguments, generator) {
  if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
  var g = generator.apply(thisArg, _arguments || []), i, q = [];
  return i = Object.create((typeof AsyncIterator === "function" ? AsyncIterator : Object).prototype), verb("next"), verb("throw"), verb("return", awaitReturn), i[Symbol.asyncIterator] = function() {
    return this;
  }, i;
  function awaitReturn(f) {
    return function(v) {
      return Promise.resolve(v).then(f, reject);
    };
  }
  function verb(n, f) {
    if (g[n]) {
      i[n] = function(v) {
        return new Promise(function(a, b) {
          q.push([n, v, a, b]) > 1 || resume(n, v);
        });
      };
      if (f) i[n] = f(i[n]);
    }
  }
  function resume(n, v) {
    try {
      step(g[n](v));
    } catch (e) {
      settle(q[0][3], e);
    }
  }
  function step(r) {
    r.value instanceof __await2 ? Promise.resolve(r.value.v).then(fulfill, reject) : settle(q[0][2], r);
  }
  function fulfill(value) {
    resume("next", value);
  }
  function reject(value) {
    resume("throw", value);
  }
  function settle(f, v) {
    if (f(v), q.shift(), q.length) resume(q[0][0], q[0][1]);
  }
};
function createClient(service, transport2) {
  return makeAnyClient(service, (method) => {
    switch (method.methodKind) {
      case "unary":
        return createUnaryFn(transport2, method);
      case "server_streaming":
        return createServerStreamingFn(transport2, method);
      case "client_streaming":
        return createClientStreamingFn(transport2, method);
      case "bidi_streaming":
        return createBiDiStreamingFn(transport2, method);
      default:
        return null;
    }
  });
}
function createUnaryFn(transport2, method) {
  return async (input2, options2) => {
    var _a, _b;
    const response = await transport2.unary(method, options2 === null || options2 === void 0 ? void 0 : options2.signal, options2 === null || options2 === void 0 ? void 0 : options2.timeoutMs, options2 === null || options2 === void 0 ? void 0 : options2.headers, input2, options2 === null || options2 === void 0 ? void 0 : options2.contextValues);
    (_a = options2 === null || options2 === void 0 ? void 0 : options2.onHeader) === null || _a === void 0 ? void 0 : _a.call(options2, response.header);
    (_b = options2 === null || options2 === void 0 ? void 0 : options2.onTrailer) === null || _b === void 0 ? void 0 : _b.call(options2, response.trailer);
    return response.message;
  };
}
function createServerStreamingFn(transport2, method) {
  return (input2, options2) => handleStreamResponse(transport2.stream(method, options2 === null || options2 === void 0 ? void 0 : options2.signal, options2 === null || options2 === void 0 ? void 0 : options2.timeoutMs, options2 === null || options2 === void 0 ? void 0 : options2.headers, createAsyncIterable([input2]), options2 === null || options2 === void 0 ? void 0 : options2.contextValues), options2);
}
function createClientStreamingFn(transport2, method) {
  return async (request, options2) => {
    var _a, e_1, _b, _c;
    var _d, _e;
    const response = await transport2.stream(method, options2 === null || options2 === void 0 ? void 0 : options2.signal, options2 === null || options2 === void 0 ? void 0 : options2.timeoutMs, options2 === null || options2 === void 0 ? void 0 : options2.headers, request, options2 === null || options2 === void 0 ? void 0 : options2.contextValues);
    (_d = options2 === null || options2 === void 0 ? void 0 : options2.onHeader) === null || _d === void 0 ? void 0 : _d.call(options2, response.header);
    let singleMessage;
    let count = 0;
    try {
      for (var _f = true, _g = __asyncValues2(response.message), _h; _h = await _g.next(), _a = _h.done, !_a; _f = true) {
        _c = _h.value;
        _f = false;
        const message = _c;
        singleMessage = message;
        count++;
      }
    } catch (e_1_1) {
      e_1 = { error: e_1_1 };
    } finally {
      try {
        if (!_f && !_a && (_b = _g.return)) await _b.call(_g);
      } finally {
        if (e_1) throw e_1.error;
      }
    }
    if (!singleMessage) {
      throw new ConnectError("protocol error: missing response message", Code.Unimplemented);
    }
    if (count > 1) {
      throw new ConnectError("protocol error: received extra messages for client streaming method", Code.Unimplemented);
    }
    (_e = options2 === null || options2 === void 0 ? void 0 : options2.onTrailer) === null || _e === void 0 ? void 0 : _e.call(options2, response.trailer);
    return singleMessage;
  };
}
function createBiDiStreamingFn(transport2, method) {
  return (request, options2) => handleStreamResponse(transport2.stream(method, options2 === null || options2 === void 0 ? void 0 : options2.signal, options2 === null || options2 === void 0 ? void 0 : options2.timeoutMs, options2 === null || options2 === void 0 ? void 0 : options2.headers, request, options2 === null || options2 === void 0 ? void 0 : options2.contextValues), options2);
}
function handleStreamResponse(stream, options2) {
  const it = (function() {
    return __asyncGenerator2(this, arguments, function* () {
      var _a, _b;
      const response = yield __await2(stream);
      (_a = options2 === null || options2 === void 0 ? void 0 : options2.onHeader) === null || _a === void 0 ? void 0 : _a.call(options2, response.header);
      yield __await2(yield* __asyncDelegator2(__asyncValues2(response.message)));
      (_b = options2 === null || options2 === void 0 ? void 0 : options2.onTrailer) === null || _b === void 0 ? void 0 : _b.call(options2, response.trailer);
    });
  })()[Symbol.asyncIterator]();
  return {
    [Symbol.asyncIterator]: () => ({
      next: () => it.next()
    })
  };
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol/signals.js
function createLinkedAbortController(...signals) {
  const controller = new AbortController();
  const sa = signals.filter((s) => s !== void 0).concat(controller.signal);
  for (const signal of sa) {
    if (signal.aborted) {
      onAbort.apply(signal);
      break;
    }
    signal.addEventListener("abort", onAbort);
  }
  function onAbort() {
    if (!controller.signal.aborted) {
      controller.abort(getAbortSignalReason(this));
    }
    for (const signal of sa) {
      signal.removeEventListener("abort", onAbort);
    }
  }
  return controller;
}
function createDeadlineSignal(timeoutMs) {
  const controller = new AbortController();
  const listener = () => {
    controller.abort(new ConnectError("the operation timed out", Code.DeadlineExceeded));
  };
  let timeoutId;
  if (timeoutMs !== void 0) {
    if (timeoutMs <= 0)
      listener();
    else
      timeoutId = setTimeout(listener, timeoutMs);
  }
  return {
    signal: controller.signal,
    cleanup: () => clearTimeout(timeoutId)
  };
}
function getAbortSignalReason(signal) {
  if (!signal.aborted) {
    return void 0;
  }
  if (signal.reason !== void 0) {
    return signal.reason;
  }
  const e = new Error("This operation was aborted");
  e.name = "AbortError";
  return e;
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/context-values.js
function createContextValues() {
  return {
    get(key) {
      return key.id in this ? this[key.id] : key.defaultValue;
    },
    set(key, value) {
      this[key.id] = value;
      return this;
    },
    delete(key) {
      delete this[key.id];
      return this;
    }
  };
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/headers.js
var headerContentType = "Content-Type";
var headerUnaryContentLength = "Content-Length";
var headerUnaryEncoding = "Content-Encoding";
var headerUnaryAcceptEncoding = "Accept-Encoding";
var headerTimeout = "Connect-Timeout-Ms";
var headerProtocolVersion = "Connect-Protocol-Version";
var headerUserAgent = "User-Agent";

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol/create-method-url.js
function createMethodUrl(baseUrl, method) {
  return baseUrl.toString().replace(/\/?$/, `/${method.parent.typeName}/${method.name}`);
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol/normalize.js
function normalize(desc, message) {
  return create(desc, message);
}
function normalizeIterable(desc, input2) {
  function transform(result) {
    if (result.done === true) {
      return result;
    }
    return {
      done: result.done,
      value: normalize(desc, result.value)
    };
  }
  return {
    [Symbol.asyncIterator]() {
      const it = input2[Symbol.asyncIterator]();
      const res = {
        next: () => it.next().then(transform)
      };
      if (it.throw !== void 0) {
        res.throw = (e) => it.throw(e).then(transform);
      }
      if (it.return !== void 0) {
        res.return = (v) => it.return(v).then(transform);
      }
      return res;
    }
  };
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/interceptor.js
function applyInterceptors(next, interceptors) {
  if (!interceptors) {
    return next;
  }
  for (const i of interceptors.concat().reverse()) {
    next = i(next);
  }
  return next;
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol/serialization.js
function getJsonOptions(options2) {
  var _a;
  const o = Object.assign({}, options2);
  (_a = o.ignoreUnknownFields) !== null && _a !== void 0 ? _a : o.ignoreUnknownFields = true;
  return o;
}
function createClientMethodSerializers(method, useBinaryFormat, jsonOptions, binaryOptions) {
  const input2 = useBinaryFormat ? createBinarySerialization(method.input, binaryOptions) : createJsonSerialization(method.input, jsonOptions);
  const output = useBinaryFormat ? createBinarySerialization(method.output, binaryOptions) : createJsonSerialization(method.output, jsonOptions);
  return { parse: output.parse, serialize: input2.serialize };
}
function createBinarySerialization(desc, options2) {
  return {
    parse(data) {
      try {
        return fromBinary(desc, data, options2);
      } catch (e) {
        const m = e instanceof Error ? e.message : String(e);
        throw new ConnectError(`parse binary: ${m}`, Code.Internal);
      }
    },
    serialize(data) {
      try {
        return toBinary(desc, data, options2);
      } catch (e) {
        const m = e instanceof Error ? e.message : String(e);
        throw new ConnectError(`serialize binary: ${m}`, Code.Internal);
      }
    }
  };
}
function createJsonSerialization(desc, options2) {
  var _a, _b;
  const textEncoder = (_a = options2 === null || options2 === void 0 ? void 0 : options2.textEncoder) !== null && _a !== void 0 ? _a : new TextEncoder();
  const textDecoder = (_b = options2 === null || options2 === void 0 ? void 0 : options2.textDecoder) !== null && _b !== void 0 ? _b : new TextDecoder();
  const o = getJsonOptions(options2);
  return {
    parse(data) {
      try {
        const json = textDecoder.decode(data);
        return fromJsonString(desc, json, o);
      } catch (e) {
        throw ConnectError.from(e, Code.InvalidArgument);
      }
    },
    serialize(data) {
      try {
        const json = toJsonString(desc, data, o);
        return textEncoder.encode(json);
      } catch (e) {
        throw ConnectError.from(e, Code.Internal);
      }
    }
  };
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/content-type.js
var contentTypeRegExp = /^application\/(connect\+)?(?:(json)(?:; ?charset=utf-?8)?|(proto))$/i;
var contentTypeUnaryProto = "application/proto";
var contentTypeUnaryJson = "application/json";
var contentTypeStreamProto = "application/connect+proto";
var contentTypeStreamJson = "application/connect+json";
function parseContentType(contentType) {
  const match = contentType === null || contentType === void 0 ? void 0 : contentType.match(contentTypeRegExp);
  if (!match) {
    return void 0;
  }
  const stream = !!match[1];
  const binary = !!match[3];
  return { stream, binary };
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/error-json.js
function errorFromJson(jsonValue, metadata, fallback) {
  var _a;
  if (metadata) {
    new Headers(metadata).forEach((value, key) => fallback.metadata.append(key, value));
  }
  if (typeof jsonValue !== "object" || jsonValue == null || Array.isArray(jsonValue)) {
    throw fallback;
  }
  let code = fallback.code;
  if ("code" in jsonValue && typeof jsonValue.code === "string") {
    code = (_a = codeFromString(jsonValue.code)) !== null && _a !== void 0 ? _a : code;
  }
  const message = jsonValue.message;
  if (message != null && typeof message !== "string") {
    throw fallback;
  }
  const error = new ConnectError(message !== null && message !== void 0 ? message : "", code, metadata);
  error.isWireError = true;
  if ("details" in jsonValue && Array.isArray(jsonValue.details)) {
    for (const detail of jsonValue.details) {
      if (detail === null || typeof detail != "object" || Array.isArray(detail) || typeof detail.type != "string" || typeof detail.value != "string") {
        throw fallback;
      }
      try {
        error.details.push({
          type: detail.type,
          value: base64Decode(detail.value),
          debug: detail.debug
        });
      } catch (e) {
        throw fallback;
      }
    }
  }
  return error;
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/end-stream.js
var endStreamFlag = 2;
function endStreamFromJson(data) {
  const parseErr = new ConnectError("invalid end stream", Code.Unknown);
  let jsonValue;
  try {
    jsonValue = JSON.parse(typeof data == "string" ? data : new TextDecoder().decode(data));
  } catch (e) {
    throw parseErr;
  }
  if (typeof jsonValue != "object" || jsonValue == null || Array.isArray(jsonValue)) {
    throw parseErr;
  }
  const metadata = new Headers();
  if ("metadata" in jsonValue) {
    if (typeof jsonValue.metadata != "object" || jsonValue.metadata == null || Array.isArray(jsonValue.metadata)) {
      throw parseErr;
    }
    for (const [key, values] of Object.entries(jsonValue.metadata)) {
      if (!Array.isArray(values) || values.some((value) => typeof value != "string")) {
        throw parseErr;
      }
      for (const value of values) {
        metadata.append(key, value);
      }
    }
  }
  const error = "error" in jsonValue && jsonValue.error != null ? errorFromJson(jsonValue.error, metadata, parseErr) : void 0;
  return { metadata, error };
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/http-status.js
function codeFromHttpStatus(httpStatus) {
  switch (httpStatus) {
    case 400:
      return Code.Internal;
    case 401:
      return Code.Unauthenticated;
    case 403:
      return Code.PermissionDenied;
    case 404:
      return Code.Unimplemented;
    case 429:
      return Code.Unavailable;
    case 502:
      return Code.Unavailable;
    case 503:
      return Code.Unavailable;
    case 504:
      return Code.Unavailable;
    default:
      return Code.Unknown;
  }
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/trailer-mux.js
function trailerDemux(header) {
  const h = new Headers(), t = new Headers();
  header.forEach((value, key) => {
    if (key.toLowerCase().startsWith("trailer-")) {
      t.append(key.substring(8), value);
    } else {
      h.append(key, value);
    }
  });
  return [h, t];
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/version.js
var protocolVersion = "1";

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/request-header.js
function requestHeader(methodKind, useBinaryFormat, timeoutMs, userProvidedHeaders, setUserAgent) {
  const result = new Headers(userProvidedHeaders !== null && userProvidedHeaders !== void 0 ? userProvidedHeaders : {});
  if (timeoutMs !== void 0) {
    result.set(headerTimeout, `${timeoutMs}`);
  }
  result.set(headerContentType, methodKind == "unary" ? useBinaryFormat ? contentTypeUnaryProto : contentTypeUnaryJson : useBinaryFormat ? contentTypeStreamProto : contentTypeStreamJson);
  result.set(headerProtocolVersion, protocolVersion);
  if (!result.has(headerUserAgent) && setUserAgent) {
    result.set(headerUserAgent, "connect-es/2.2.0");
  }
  return result;
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/validate-response.js
function validateResponse(methodKind, useBinaryFormat, status2, headers) {
  const mimeType = headers.get(headerContentType);
  const parsedType = parseContentType(mimeType);
  if (status2 !== 200) {
    const errorFromStatus = new ConnectError(`HTTP ${status2}`, codeFromHttpStatus(status2), headers);
    if (methodKind == "unary" && parsedType && !parsedType.binary) {
      return { isUnaryError: true, unaryError: errorFromStatus };
    }
    throw errorFromStatus;
  }
  const allowedContentType = {
    binary: useBinaryFormat,
    stream: methodKind !== "unary"
  };
  if ((parsedType === null || parsedType === void 0 ? void 0 : parsedType.binary) !== allowedContentType.binary || parsedType.stream !== allowedContentType.stream) {
    throw new ConnectError(`unsupported content type ${mimeType}`, parsedType === void 0 ? Code.Unknown : Code.Internal, headers);
  }
  return { isUnaryError: false };
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/get-request.js
var contentTypePrefix = "application/";
function encodeMessageForUrl(message, useBase64) {
  if (useBase64) {
    return base64Encode(message, "url");
  }
  return encodeURIComponent(new TextDecoder().decode(message));
}
function transformConnectPostToGetRequest(request, message, useBase64) {
  let query = `?connect=v${protocolVersion}`;
  let compressionQuery = "";
  const compression = request.header.get(headerUnaryEncoding);
  if (compression !== null && compression !== "identity") {
    compressionQuery = "&compression=" + encodeURIComponent(compression);
    useBase64 = true;
  }
  if (useBase64) {
    query += "&base64=1";
  }
  query += compressionQuery;
  const contentType = request.header.get(headerContentType);
  if ((contentType === null || contentType === void 0 ? void 0 : contentType.indexOf(contentTypePrefix)) === 0) {
    query += "&encoding=" + encodeURIComponent(contentType.slice(contentTypePrefix.length));
  }
  query += "&message=" + encodeMessageForUrl(message, useBase64);
  const url = request.url + query;
  const header = new Headers(request.header);
  for (const h of [
    headerProtocolVersion,
    headerContentType,
    headerUnaryContentLength,
    headerUnaryEncoding,
    headerUnaryAcceptEncoding
  ]) {
    header.delete(h);
  }
  return Object.assign(Object.assign({}, request), {
    requestMethod: "GET",
    url,
    header
  });
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol/run-call.js
function runUnaryCall(opt) {
  const next = applyInterceptors(opt.next, opt.interceptors);
  const [signal, abort, done] = setupSignal(opt);
  const req = Object.assign(Object.assign({}, opt.req), { message: normalize(opt.req.method.input, opt.req.message), signal });
  return next(req).then((res) => {
    done();
    return res;
  }, abort);
}
function runStreamingCall(opt) {
  const next = applyInterceptors(opt.next, opt.interceptors);
  const [signal, abort, done] = setupSignal(opt);
  const req = Object.assign(Object.assign({}, opt.req), { message: normalizeIterable(opt.req.method.input, opt.req.message), signal });
  let doneCalled = false;
  signal.addEventListener("abort", function() {
    var _a, _b;
    const it = opt.req.message[Symbol.asyncIterator]();
    if (!doneCalled) {
      (_a = it.throw) === null || _a === void 0 ? void 0 : _a.call(it, this.reason).catch(() => {
      });
    }
    (_b = it.return) === null || _b === void 0 ? void 0 : _b.call(it).catch(() => {
    });
  });
  return next(req).then((res) => {
    return Object.assign(Object.assign({}, res), { message: {
      [Symbol.asyncIterator]() {
        const it = res.message[Symbol.asyncIterator]();
        return {
          next() {
            if (!doneCalled && signal.aborted) {
              return abort(getAbortSignalReason(signal));
            }
            return it.next().then((r) => {
              if (r.done == true) {
                doneCalled = true;
                done();
              }
              return r;
            }, abort);
          }
          // We deliberately omit throw/return.
        };
      }
    } });
  }, abort);
}
function setupSignal(opt) {
  const { signal, cleanup } = createDeadlineSignal(opt.timeoutMs);
  const controller = createLinkedAbortController(opt.signal, signal);
  return [
    controller.signal,
    function abort(reason) {
      const e = controller.signal.aborted ? ConnectError.from(getAbortSignalReason(controller.signal), Code.Canceled) : ConnectError.from(reason);
      controller.abort(e);
      cleanup();
      return Promise.reject(e);
    },
    function done() {
      cleanup();
      controller.abort();
    }
  ];
}

// demo/web/node_modules/@connectrpc/connect-web/dist/esm/assert-fetch-api.js
function assertFetchApi() {
  try {
    new Headers();
  } catch (_) {
    throw new Error("connect-web requires the fetch API. Are you running on an old version of Node.js? Node.js is not supported in Connect for Web - please stay tuned for Connect for Node.");
  }
}

// demo/web/node_modules/@connectrpc/connect-web/dist/esm/connect-transport.js
var __await3 = function(v) {
  return this instanceof __await3 ? (this.v = v, this) : new __await3(v);
};
var __asyncGenerator3 = function(thisArg, _arguments, generator) {
  if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
  var g = generator.apply(thisArg, _arguments || []), i, q = [];
  return i = Object.create((typeof AsyncIterator === "function" ? AsyncIterator : Object).prototype), verb("next"), verb("throw"), verb("return", awaitReturn), i[Symbol.asyncIterator] = function() {
    return this;
  }, i;
  function awaitReturn(f) {
    return function(v) {
      return Promise.resolve(v).then(f, reject);
    };
  }
  function verb(n, f) {
    if (g[n]) {
      i[n] = function(v) {
        return new Promise(function(a, b) {
          q.push([n, v, a, b]) > 1 || resume(n, v);
        });
      };
      if (f) i[n] = f(i[n]);
    }
  }
  function resume(n, v) {
    try {
      step(g[n](v));
    } catch (e) {
      settle(q[0][3], e);
    }
  }
  function step(r) {
    r.value instanceof __await3 ? Promise.resolve(r.value.v).then(fulfill, reject) : settle(q[0][2], r);
  }
  function fulfill(value) {
    resume("next", value);
  }
  function reject(value) {
    resume("throw", value);
  }
  function settle(f, v) {
    if (f(v), q.shift(), q.length) resume(q[0][0], q[0][1]);
  }
};
var fetchOptions = {
  redirect: "error"
};
function createConnectTransport(options2) {
  var _a;
  assertFetchApi();
  const useBinaryFormat = (_a = options2.useBinaryFormat) !== null && _a !== void 0 ? _a : false;
  return {
    async unary(method, signal, timeoutMs, header, message, contextValues) {
      const { serialize, parse } = createClientMethodSerializers(method, useBinaryFormat, options2.jsonOptions, options2.binaryOptions);
      timeoutMs = timeoutMs === void 0 ? options2.defaultTimeoutMs : timeoutMs <= 0 ? void 0 : timeoutMs;
      return await runUnaryCall({
        interceptors: options2.interceptors,
        signal,
        timeoutMs,
        req: {
          stream: false,
          service: method.parent,
          method,
          requestMethod: "POST",
          url: createMethodUrl(options2.baseUrl, method),
          header: requestHeader(method.methodKind, useBinaryFormat, timeoutMs, header, false),
          contextValues: contextValues !== null && contextValues !== void 0 ? contextValues : createContextValues(),
          message
        },
        next: async (req) => {
          var _a2;
          const useGet = options2.useHttpGet === true && method.idempotency === MethodOptions_IdempotencyLevel.NO_SIDE_EFFECTS;
          let body2 = null;
          if (useGet) {
            req = transformConnectPostToGetRequest(req, serialize(req.message), useBinaryFormat);
          } else {
            body2 = serialize(req.message);
          }
          const fetch2 = (_a2 = options2.fetch) !== null && _a2 !== void 0 ? _a2 : globalThis.fetch;
          const response = await fetch2(req.url, Object.assign(Object.assign({}, fetchOptions), { method: req.requestMethod, headers: req.header, signal: req.signal, body: body2 }));
          const { isUnaryError, unaryError } = validateResponse(method.methodKind, useBinaryFormat, response.status, response.headers);
          if (isUnaryError) {
            throw errorFromJson(await response.json(), appendHeaders(...trailerDemux(response.headers)), unaryError);
          }
          const [demuxedHeader, demuxedTrailer] = trailerDemux(response.headers);
          return {
            stream: false,
            service: method.parent,
            method,
            header: demuxedHeader,
            message: useBinaryFormat ? parse(new Uint8Array(await response.arrayBuffer())) : fromJson(method.output, await response.json(), getJsonOptions(options2.jsonOptions)),
            trailer: demuxedTrailer
          };
        }
      });
    },
    async stream(method, signal, timeoutMs, header, input2, contextValues) {
      const { serialize, parse } = createClientMethodSerializers(method, useBinaryFormat, options2.jsonOptions, options2.binaryOptions);
      function parseResponseBody(body2, trailerTarget, header2, signal2) {
        return __asyncGenerator3(this, arguments, function* parseResponseBody_1() {
          const reader = createEnvelopeReadableStream(body2).getReader();
          let endStreamReceived = false;
          for (; ; ) {
            const result = yield __await3(reader.read());
            if (result.done) {
              break;
            }
            const { flags, data } = result.value;
            if ((flags & compressedFlag) === compressedFlag) {
              throw new ConnectError(`protocol error: received unsupported compressed output`, Code.Internal);
            }
            if ((flags & endStreamFlag) === endStreamFlag) {
              endStreamReceived = true;
              const endStream = endStreamFromJson(data);
              if (endStream.error) {
                const error = endStream.error;
                header2.forEach((value, key) => {
                  error.metadata.append(key, value);
                });
                throw error;
              }
              endStream.metadata.forEach((value, key) => trailerTarget.set(key, value));
              continue;
            }
            yield yield __await3(parse(data));
          }
          if ("throwIfAborted" in signal2) {
            signal2.throwIfAborted();
          }
          if (!endStreamReceived) {
            throw "missing EndStreamResponse";
          }
        });
      }
      async function createRequestBody(input3) {
        if (method.methodKind != "server_streaming") {
          throw "The fetch API does not support streaming request bodies";
        }
        const r = await input3[Symbol.asyncIterator]().next();
        if (r.done == true) {
          throw "missing request message";
        }
        return encodeEnvelope(0, serialize(r.value));
      }
      timeoutMs = timeoutMs === void 0 ? options2.defaultTimeoutMs : timeoutMs <= 0 ? void 0 : timeoutMs;
      return await runStreamingCall({
        interceptors: options2.interceptors,
        timeoutMs,
        signal,
        req: {
          stream: true,
          service: method.parent,
          method,
          requestMethod: "POST",
          url: createMethodUrl(options2.baseUrl, method),
          header: requestHeader(method.methodKind, useBinaryFormat, timeoutMs, header, false),
          contextValues: contextValues !== null && contextValues !== void 0 ? contextValues : createContextValues(),
          message: input2
        },
        next: async (req) => {
          var _a2;
          const fetch2 = (_a2 = options2.fetch) !== null && _a2 !== void 0 ? _a2 : globalThis.fetch;
          const fRes = await fetch2(req.url, Object.assign(Object.assign({}, fetchOptions), { method: req.requestMethod, headers: req.header, signal: req.signal, body: await createRequestBody(req.message) }));
          validateResponse(method.methodKind, useBinaryFormat, fRes.status, fRes.headers);
          if (fRes.body === null) {
            throw "missing response body";
          }
          const trailer = new Headers();
          const res = Object.assign(Object.assign({}, req), { header: fRes.headers, trailer, message: parseResponseBody(fRes.body, trailer, fRes.headers, req.signal) });
          return res;
        }
      });
    }
  };
}

// demo/web/src/gen/open_inference_grpc_pb.ts
var file_open_inference_grpc = /* @__PURE__ */ fileDesc("ChlvcGVuX2luZmVyZW5jZV9ncnBjLnByb3RvEglpbmZlcmVuY2UiEwoRU2VydmVyTGl2ZVJlcXVlc3QiIgoSU2VydmVyTGl2ZVJlc3BvbnNlEgwKBGxpdmUYASABKAgiFAoSU2VydmVyUmVhZHlSZXF1ZXN0IiQKE1NlcnZlclJlYWR5UmVzcG9uc2USDQoFcmVhZHkYASABKAgiMgoRTW9kZWxSZWFkeVJlcXVlc3QSDAoEbmFtZRgBIAEoCRIPCgd2ZXJzaW9uGAIgASgJIiMKEk1vZGVsUmVhZHlSZXNwb25zZRINCgVyZWFkeRgBIAEoCCIXChVTZXJ2ZXJNZXRhZGF0YVJlcXVlc3QiSwoWU2VydmVyTWV0YWRhdGFSZXNwb25zZRIMCgRuYW1lGAEgASgJEg8KB3ZlcnNpb24YAiABKAkSEgoKZXh0ZW5zaW9ucxgDIAMoCSI1ChRNb2RlbE1ldGFkYXRhUmVxdWVzdBIMCgRuYW1lGAEgASgJEg8KB3ZlcnNpb24YAiABKAkihgMKFU1vZGVsTWV0YWRhdGFSZXNwb25zZRIMCgRuYW1lGAEgASgJEhAKCHZlcnNpb25zGAIgAygJEhAKCHBsYXRmb3JtGAMgASgJEj8KBmlucHV0cxgEIAMoCzIvLmluZmVyZW5jZS5Nb2RlbE1ldGFkYXRhUmVzcG9uc2UuVGVuc29yTWV0YWRhdGESQAoHb3V0cHV0cxgFIAMoCzIvLmluZmVyZW5jZS5Nb2RlbE1ldGFkYXRhUmVzcG9uc2UuVGVuc29yTWV0YWRhdGESRAoKcHJvcGVydGllcxgGIAMoCzIwLmluZmVyZW5jZS5Nb2RlbE1ldGFkYXRhUmVzcG9uc2UuUHJvcGVydGllc0VudHJ5Gj8KDlRlbnNvck1ldGFkYXRhEgwKBG5hbWUYASABKAkSEAoIZGF0YXR5cGUYAiABKAkSDQoFc2hhcGUYAyADKAMaMQoPUHJvcGVydGllc0VudHJ5EgsKA2tleRgBIAEoCRINCgV2YWx1ZRgCIAEoCToCOAEi7gYKEU1vZGVsSW5mZXJSZXF1ZXN0EhIKCm1vZGVsX25hbWUYASABKAkSFQoNbW9kZWxfdmVyc2lvbhgCIAEoCRIKCgJpZBgDIAEoCRJACgpwYXJhbWV0ZXJzGAQgAygLMiwuaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXF1ZXN0LlBhcmFtZXRlcnNFbnRyeRI9CgZpbnB1dHMYBSADKAsyLS5pbmZlcmVuY2UuTW9kZWxJbmZlclJlcXVlc3QuSW5mZXJJbnB1dFRlbnNvchJICgdvdXRwdXRzGAYgAygLMjcuaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXF1ZXN0LkluZmVyUmVxdWVzdGVkT3V0cHV0VGVuc29yEhoKEnJhd19pbnB1dF9jb250ZW50cxgHIAMoDBqUAgoQSW5mZXJJbnB1dFRlbnNvchIMCgRuYW1lGAEgASgJEhAKCGRhdGF0eXBlGAIgASgJEg0KBXNoYXBlGAMgAygDElEKCnBhcmFtZXRlcnMYBCADKAsyPS5pbmZlcmVuY2UuTW9kZWxJbmZlclJlcXVlc3QuSW5mZXJJbnB1dFRlbnNvci5QYXJhbWV0ZXJzRW50cnkSMAoIY29udGVudHMYBSABKAsyHi5pbmZlcmVuY2UuSW5mZXJUZW5zb3JDb250ZW50cxpMCg9QYXJhbWV0ZXJzRW50cnkSCwoDa2V5GAEgASgJEigKBXZhbHVlGAIgASgLMhkuaW5mZXJlbmNlLkluZmVyUGFyYW1ldGVyOgI4ARrVAQoaSW5mZXJSZXF1ZXN0ZWRPdXRwdXRUZW5zb3ISDAoEbmFtZRgBIAEoCRJbCgpwYXJhbWV0ZXJzGAIgAygLMkcuaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXF1ZXN0LkluZmVyUmVxdWVzdGVkT3V0cHV0VGVuc29yLlBhcmFtZXRlcnNFbnRyeRpMCg9QYXJhbWV0ZXJzRW50cnkSCwoDa2V5GAEgASgJEigKBXZhbHVlGAIgASgLMhkuaW5mZXJlbmNlLkluZmVyUGFyYW1ldGVyOgI4ARpMCg9QYXJhbWV0ZXJzRW50cnkSCwoDa2V5GAEgASgJEigKBXZhbHVlGAIgASgLMhkuaW5mZXJlbmNlLkluZmVyUGFyYW1ldGVyOgI4ASLVBAoSTW9kZWxJbmZlclJlc3BvbnNlEhIKCm1vZGVsX25hbWUYASABKAkSFQoNbW9kZWxfdmVyc2lvbhgCIAEoCRIKCgJpZBgDIAEoCRJBCgpwYXJhbWV0ZXJzGAQgAygLMi0uaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXNwb25zZS5QYXJhbWV0ZXJzRW50cnkSQAoHb3V0cHV0cxgFIAMoCzIvLmluZmVyZW5jZS5Nb2RlbEluZmVyUmVzcG9uc2UuSW5mZXJPdXRwdXRUZW5zb3ISGwoTcmF3X291dHB1dF9jb250ZW50cxgGIAMoDBqXAgoRSW5mZXJPdXRwdXRUZW5zb3ISDAoEbmFtZRgBIAEoCRIQCghkYXRhdHlwZRgCIAEoCRINCgVzaGFwZRgDIAMoAxJTCgpwYXJhbWV0ZXJzGAQgAygLMj8uaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXNwb25zZS5JbmZlck91dHB1dFRlbnNvci5QYXJhbWV0ZXJzRW50cnkSMAoIY29udGVudHMYBSABKAsyHi5pbmZlcmVuY2UuSW5mZXJUZW5zb3JDb250ZW50cxpMCg9QYXJhbWV0ZXJzRW50cnkSCwoDa2V5GAEgASgJEigKBXZhbHVlGAIgASgLMhkuaW5mZXJlbmNlLkluZmVyUGFyYW1ldGVyOgI4ARpMCg9QYXJhbWV0ZXJzRW50cnkSCwoDa2V5GAEgASgJEigKBXZhbHVlGAIgASgLMhkuaW5mZXJlbmNlLkluZmVyUGFyYW1ldGVyOgI4ASKZAQoOSW5mZXJQYXJhbWV0ZXISFAoKYm9vbF9wYXJhbRgBIAEoCEgAEhUKC2ludDY0X3BhcmFtGAIgASgDSAASFgoMc3RyaW5nX3BhcmFtGAMgASgJSAASFgoMZG91YmxlX3BhcmFtGAQgASgBSAASFgoMdWludDY0X3BhcmFtGAUgASgESABCEgoQcGFyYW1ldGVyX2Nob2ljZSLQAQoTSW5mZXJUZW5zb3JDb250ZW50cxIVCg1ib29sX2NvbnRlbnRzGAEgAygIEhQKDGludF9jb250ZW50cxgCIAMoBRIWCg5pbnQ2NF9jb250ZW50cxgDIAMoAxIVCg11aW50X2NvbnRlbnRzGAQgAygNEhcKD3VpbnQ2NF9jb250ZW50cxgFIAMoBBIVCg1mcDMyX2NvbnRlbnRzGAYgAygCEhUKDWZwNjRfY29udGVudHMYByADKAESFgoOYnl0ZXNfY29udGVudHMYCCADKAwy/AMKFEdSUENJbmZlcmVuY2VTZXJ2aWNlEksKClNlcnZlckxpdmUSHC5pbmZlcmVuY2UuU2VydmVyTGl2ZVJlcXVlc3QaHS5pbmZlcmVuY2UuU2VydmVyTGl2ZVJlc3BvbnNlIgASTgoLU2VydmVyUmVhZHkSHS5pbmZlcmVuY2UuU2VydmVyUmVhZHlSZXF1ZXN0Gh4uaW5mZXJlbmNlLlNlcnZlclJlYWR5UmVzcG9uc2UiABJLCgpNb2RlbFJlYWR5EhwuaW5mZXJlbmNlLk1vZGVsUmVhZHlSZXF1ZXN0Gh0uaW5mZXJlbmNlLk1vZGVsUmVhZHlSZXNwb25zZSIAElcKDlNlcnZlck1ldGFkYXRhEiAuaW5mZXJlbmNlLlNlcnZlck1ldGFkYXRhUmVxdWVzdBohLmluZmVyZW5jZS5TZXJ2ZXJNZXRhZGF0YVJlc3BvbnNlIgASVAoNTW9kZWxNZXRhZGF0YRIfLmluZmVyZW5jZS5Nb2RlbE1ldGFkYXRhUmVxdWVzdBogLmluZmVyZW5jZS5Nb2RlbE1ldGFkYXRhUmVzcG9uc2UiABJLCgpNb2RlbEluZmVyEhwuaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXF1ZXN0Gh0uaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXNwb25zZSIAYgZwcm90bzM");
var InferParameterSchema = /* @__PURE__ */ messageDesc(file_open_inference_grpc, 12);
var GRPCInferenceService = /* @__PURE__ */ serviceDesc(file_open_inference_grpc, 0);

// demo/web/src/gen/turbo_analysis_pb.ts
var file_turbo_analysis = /* @__PURE__ */ fileDesc("ChR0dXJib19hbmFseXNpcy5wcm90bxIPdHVyYm8ucmVzdWx0LnYxIhUKE0NhcGFiaWxpdGllc1JlcXVlc3Qi4gEKFENhcGFiaWxpdGllc1Jlc3BvbnNlEjQKB2J1bmRsZXMYASADKAsyIy50dXJiby5yZXN1bHQudjEuQnVuZGxlQ2FwYWJpbGl0aWVzEi0KBmxpbWl0cxgCIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5SZXN1bHRMaW1pdHMSLQoGc2VydmVyGAMgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNlcnZlckxpbWl0cxI2Cghwcm9maWxlcxgEIAMoCzIkLnR1cmJvLnJlc3VsdC52MS5Qcm9maWxlQ2FwYWJpbGl0aWVzIm0KE1Byb2ZpbGVDYXBhYmlsaXRpZXMSDAoEbmFtZRgBIAEoCRIPCgd2ZXJzaW9uGAIgASgJEg0KBXN0ZXBzGAMgAygJEg4KBmxheWVycxgEIAMoCRIYChByZXF1aXJlZF9idW5kbGVzGAUgAygJIngKDFNlcnZlckxpbWl0cxIeChZtYXhfc2Vzc2lvbnNfcGVyX21vZGVsGAEgASgNEhkKEW1heF9sb2FkZWRfbW9kZWxzGAIgASgNEi0KBnN0cmVhbRgDIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5TdHJlYW1MaW1pdHMi1AEKEkJ1bmRsZUNhcGFiaWxpdGllcxIMCgRuYW1lGAEgASgJEigKBW1vZGVsGAIgASgLMhkudHVyYm8ucmVzdWx0LnYxLk1vZGVsUmVmEgwKBHRhc2sYAyABKAkSDQoFcm9sZXMYBCADKAkSDgoGd2lkdGhzGAUgAygNEhUKDWNvbnRleHRfbGltaXQYBiABKA0SEwoLYmF0Y2hfbGltaXQYByABKA0SLQoGc2NoZW1hGAggASgLMh0udHVyYm8ucmVzdWx0LnYxLlNjaGVtYUxpbWl0cyKiAgoMU2NoZW1hTGltaXRzEhgKEG1heF9lbnRpdHlfdHlwZXMYASABKA0SIAoYbWF4X2NsYXNzaWZpY2F0aW9uX3Rhc2tzGAUgASgNEhsKE21heF9sYWJlbHNfcGVyX3Rhc2sYBiABKA0SFgoObWF4X3N0cnVjdHVyZXMYByABKA0SIAoYbWF4X2ZpZWxkc19wZXJfc3RydWN0dXJlGAggASgNEhYKDm1heF90ZXh0X2J5dGVzGAkgASgNEhsKE21heF9wcm9tcHRfc3Vid29yZHMYCiABKA0SEQoJbWF4X3dvcmRzGAsgASgNSgQIAhADSgQIAxAESgQIBBAFUg5tYXhfbGFiZWxfc2V0c1IKbWF4X2ZpZWxkc1IJbWF4X2RlcHRoIr8CCgxSZXN1bHRMaW1pdHMSGgoSbWF4X3NwYXJzZV9lbnRyaWVzGAEgASgEEh4KFm1heF9zcGFyc2Vfb2NjdXJyZW5jZXMYAiABKAQSHgoWbWF4X3Rva2VuX3ZlY3Rvcl9ieXRlcxgDIAEoBBIYChBtYXhfcmVyYW5rX2l0ZW1zGAQgASgEEhMKC21heF9hbnN3ZXJzGAUgASgEEhwKFG1heF9hbnN3ZXJfZnJhZ21lbnRzGAYgASgEEhMKC21heF93aW5kb3dzGAcgASgEEhIKCm1heF9jaHVua3MYCCABKAQSEgoKbWF4X2xldmVscxgJIAEoBBIVCg1tYXhfZnJhZ21lbnRzGAogASgEEhcKD21heF9ncmFwaF9ieXRlcxgLIAEoBBIZChFtYXhfcmVjb3JkX3ZhbHVlcxgMIAEoBCKhAQoRTG9hZEJ1bmRsZVJlcXVlc3QSDAoEbmFtZRgBIAEoCRIOCgZidW5kbGUYAiABKAkSDgoGZGV2aWNlGAMgASgNEhAKCHNlc3Npb25zGAQgASgNEhEKCXByZWNpc2lvbhgFIAEoDRIRCgltYXhfYmF0Y2gYBiABKA0SDwoHbWF4X3NlcRgHIAEoDRIVCg1zZWxlY3RfZGV2aWNlGAsgASgIIk8KEkxvYWRCdW5kbGVSZXNwb25zZRI5CgxjYXBhYmlsaXRpZXMYASABKAsyIy50dXJiby5yZXN1bHQudjEuQnVuZGxlQ2FwYWJpbGl0aWVzIiMKE1VubG9hZEJ1bmRsZVJlcXVlc3QSDAoEbmFtZRgBIAEoCSIWChRVbmxvYWRCdW5kbGVSZXNwb25zZSKQAgoRU3BhcnNlVGVybXNSZXN1bHQSKAoFbW9kZWwYASABKAsyGS50dXJiby5yZXN1bHQudjEuTW9kZWxSZWYSCwoDcm93GAIgAygNEgwKBHRlcm0YAyADKA0SDgoGd2VpZ2h0GAQgAygCEhAKCGhhc19zcGFuGAUgAygIEhIKCnNwYW5fc3RhcnQYBiADKA0SEAoIc3Bhbl9lbmQYByADKA0SEQoJb2NjX3N0YXJ0GAggAygNEg8KB29jY19lbmQYCSADKA0SDgoGb2NjX2F0GAogAygNEgwKBHRleHQYCyABKAkSLAoHb2Zmc2V0cxgMIAEoDjIbLnR1cmJvLnJlc3VsdC52MS5PZmZzZXRNb2RlItwBChJUb2tlblZlY3RvcnNSZXN1bHQSKAoFbW9kZWwYASABKAsyGS50dXJiby5yZXN1bHQudjEuTW9kZWxSZWYSCwoDZGltGAIgASgNEgwKBGRhdGEYAyABKAwSDQoFZW1wdHkYBCADKA0SDAoEdmlldxgHIAEoDRIUCgx3aW5kb3dfc3RhcnQYBSADKA0SEgoKd2luZG93X2VuZBgGIAMoDRIMCgR0ZXh0GAggASgJEiwKB29mZnNldHMYCSABKA4yGy50dXJiby5yZXN1bHQudjEuT2Zmc2V0TW9kZSKEAQoSUmVyYW5rU2NvcmVzUmVzdWx0EigKBW1vZGVsGAEgASgLMhkudHVyYm8ucmVzdWx0LnYxLk1vZGVsUmVmEg4KBnNvdXJjZRgCIAEoCRIXCg9zb3VyY2VfaW5zdGFuY2UYBSABKAkSDAoEaXRlbRgDIAMoDRINCgVzY29yZRgEIAMoAiK1AgoQRXh0cmFjdGlvblJlc3VsdBIoCgVtb2RlbBgBIAEoCzIZLnR1cmJvLnJlc3VsdC52MS5Nb2RlbFJlZhIMCgR2aWV3GAIgASgNEhIKCnNwYW5fc3RhcnQYAyADKA0SEAoIc3Bhbl9lbmQYBCADKA0SDQoFc2NvcmUYBSADKAESEwoLcHJvYmFiaWxpdHkYBiADKAESEgoKbnVsbF9zY29yZRgHIAEoARIYChBudWxsX3Byb2JhYmlsaXR5GAggASgBEhIKCmZyYWdfc3RhcnQYCSADKA0SEAoIZnJhZ19lbmQYCiADKA0SDwoHZnJhZ19hdBgLIAMoDRIMCgR0ZXh0GAwgASgJEiwKB29mZnNldHMYDSABKA4yGy50dXJiby5yZXN1bHQudjEuT2Zmc2V0TW9kZSLlBAoNUmVjb3Jkc1Jlc3VsdBIoCgVtb2RlbBgBIAEoCzIZLnR1cmJvLnJlc3VsdC52MS5Nb2RlbFJlZhIMCgR0ZXh0GAIgASgJEgwKBHZpZXcYAyABKA0SLAoHb2Zmc2V0cxgEIAEoDjIbLnR1cmJvLnJlc3VsdC52MS5PZmZzZXRNb2RlEhIKCnR5cGVzX25hbWUYBSABKAkSDQoFdHlwZXMYBiADKAkSEgoKcGF0aHNfbmFtZRgHIAEoCRINCgVwYXRocxgIIAMoCRIUCgxjaG9pY2VzX25hbWUYCSABKAkSDwoHY2hvaWNlcxgKIAMoCRITCgtyZWNvcmRfdHlwZRgLIAMoDRIXCg9yZWNvcmRfaW5zdGFuY2UYDCADKA0SGAoQdmFsdWVfZnJhZ19zdGFydBgNIAMoDRIWCg52YWx1ZV9mcmFnX2VuZBgOIAMoDRIVCg12YWx1ZV9mcmFnX2F0GA8gAygNEhQKDHZhbHVlX3JlY29yZBgQIAMoDRISCgp2YWx1ZV9wYXRoGBEgAygNEhMKC3ZhbHVlX3Njb3JlGBIgAygBEhMKC3ZhbHVlX2xvZ2l0GBMgAygBEhMKC3N1cHBvcnRfZW5kGBQgAygNEhYKDnN1cHBvcnRfc291cmNlGBUgAygNEhMKC3N1cHBvcnRfcm93GBYgAygNEhUKDWNob2ljZV9yZWNvcmQYFyADKA0SEwoLY2hvaWNlX3BhdGgYGCADKA0SDgoGY2hvaWNlGBkgAygNEhQKDGNob2ljZV9zY29yZRgaIAMoARIUCgxjaG9pY2VfbG9naXQYGyADKAEijAEKFEFuYWx5emVTdHJlYW1SZXF1ZXN0EjYKBmhlYWRlchgBIAEoCzIkLnR1cmJvLnJlc3VsdC52MS5BbmFseXplU3RyZWFtSGVhZGVySAASNAoIZG9jdW1lbnQYAiABKAsyIC50dXJiby5yZXN1bHQudjEuQW5hbHl6ZURvY3VtZW50SABCBgoEa2luZCJyChNBbmFseXplU3RyZWFtSGVhZGVyEiwKBW9yZGVyGAEgASgOMh0udHVyYm8ucmVzdWx0LnYxLkFuYWx5emVPcmRlchIUCgxtYXhfaW5mbGlnaHQYAiABKA0SFwoPYmF0Y2hfbGluZ2VyX21zGAMgASgNIsUBCg5BbmFseXplT3B0aW9ucxIQCgh0cnVuY2F0ZRgBIAEoDRISCgptYXhfdG9rZW5zGAIgASgNEhMKC3Byb21wdF9yb2xlGAMgASgNEhEKCW5vcm1hbGl6ZRgEIAEoDRIPCgdwb29saW5nGAUgASgNEhIKCm91dHB1dF9kaW0YBiABKA0SFAoMc3BhcnNlX3RvcF9rGAcgASgNEhQKDHNwYXJzZV9mbG9vchgIIAEoAhIUCgx0b2tlbl9zdHJpZGUYCSABKA0iygEKD0FuYWx5emVEb2N1bWVudBIKCgJpZBgBIAEoCRINCgVtb2RlbBgCIAEoCRIMCgR0ZXh0GAMgASgMEjAKB29wdGlvbnMYBCABKAsyHy50dXJiby5yZXN1bHQudjEuQW5hbHl6ZU9wdGlvbnMSLgoGb3V0cHV0GAUgASgOMh4udHVyYm8ucmVzdWx0LnYxLkFuYWx5emVPdXRwdXQSLAoHb2Zmc2V0cxgGIAEoDjIbLnR1cmJvLnJlc3VsdC52MS5PZmZzZXRNb2RlIpACChVBbmFseXplU3RyZWFtUmVzcG9uc2USCgoCaWQYASABKAkSMgoHdmVjdG9ycxgCIAEoCzIfLnR1cmJvLnJlc3VsdC52MS5BbmFseXplVmVjdG9yc0gAEjIKB2ZhaWx1cmUYAyABKAsyHy50dXJiby5yZXN1bHQudjEuQW5hbHl6ZUZhaWx1cmVIABI6CgxzcGFyc2VfdGVybXMYBCABKAsyIi50dXJiby5yZXN1bHQudjEuU3BhcnNlVGVybXNSZXN1bHRIABI8Cg10b2tlbl92ZWN0b3JzGAUgASgLMiMudHVyYm8ucmVzdWx0LnYxLlRva2VuVmVjdG9yc1Jlc3VsdEgAQgkKB291dGNvbWUiSwoOQW5hbHl6ZVZlY3RvcnMSDQoFbW9kZWwYASABKAkSDwoHdmVyc2lvbhgCIAEoCRILCgNkaW0YAyABKA0SDAoEZGF0YRgEIAEoDCJECg5BbmFseXplRmFpbHVyZRISCgp0dXJib19jb2RlGAEgASgFEg0KBWZpZWxkGAIgASgNEg8KB21lc3NhZ2UYAyABKAki/wEKDFN0cmVhbUxpbWl0cxIjChttYXhfcXVldWVkX2l0ZW1zX3Blcl9zdHJlYW0YASABKAQSIwobbWF4X3F1ZXVlZF9ieXRlc19wZXJfc3RyZWFtGAIgASgEEh8KF21heF9xdWV1ZWRfaXRlbXNfc2VydmVyGAMgASgEEh8KF21heF9xdWV1ZWRfYnl0ZXNfc2VydmVyGAQgASgEEiUKHW1heF9pbmZsaWdodF9pdGVtc19wZXJfc3RyZWFtGAUgASgEEh8KF2RlZmF1bHRfYmF0Y2hfbGluZ2VyX21zGAYgASgEEhsKE21heF9iYXRjaF9saW5nZXJfbXMYByABKAQi5wEKCUxheWVySW5mbxINCgVsYXllchgBIAEoCRIQCghpbnN0YW5jZRgCIAEoCRIwCgZzdGF0dXMYAyABKA4yIC50dXJiby5yZXN1bHQudjEuTGF5ZXJXaXJlU3RhdHVzEhUKDXN0YXR1c19kZXRhaWwYBCABKAkSEQoJYW5ub3RhdG9yGAUgASgJEg8KB3ZlcnNpb24YBiABKAkSKAoFbW9kZWwYByABKAsyGS50dXJiby5yZXN1bHQudjEuTW9kZWxSZWYSEwoLY29uZmlnX2hhc2gYCCABKAQSDQoFaXRlbXMYCSABKA0iJQoURG9jdW1lbnRHcmFwaFJlcXVlc3QSDQoFZ3JhcGgYASABKAwikQEKFURvY3VtZW50R3JhcGhSZXNwb25zZRINCgVncmFwaBgBIAEoDBIqCgZsYXllcnMYAiADKAsyGi50dXJiby5yZXN1bHQudjEuTGF5ZXJJbmZvEigKBXZpZXdzGAMgAygLMhkudHVyYm8ucmVzdWx0LnYxLlZpZXdJbmZvEhMKC3NvdXJjZV90ZXh0GAQgASgJItsEChFDaHVua0xldmVsc1Jlc3VsdBIMCgR0ZXh0GAEgASgJEgwKBHZpZXcYAiABKA0SLAoHb2Zmc2V0cxgDIAEoDjIbLnR1cmJvLnJlc3VsdC52MS5PZmZzZXRNb2RlEhgKEGNodW5rX2ZyYWdfc3RhcnQYBCADKA0SFgoOY2h1bmtfZnJhZ19lbmQYBSADKA0SFQoNY2h1bmtfZnJhZ19hdBgGIAMoDRITCgtjaHVua19sZXZlbBgHIAMoDRIUCgxjaHVua19wYXJlbnQYCCADKAUSHAoUY2h1bmtfc2VudGVuY2Vfc3RhcnQYCSADKA0SGgoSY2h1bmtfc2VudGVuY2VfZW5kGAogAygNEhEKCWNodW5rX2N1dBgLIAMoDRIXCg9jaHVua19jdXRfYmxvY2sYDCADKA0SEgoKY2h1bmtfY29zdBgNIAMoAxIaChJjaHVua19tb2RlbF90b2tlbnMYGCADKA0SGQoRY2h1bmtfb3Zlcl9idWRnZXQYGSADKAgSGAoQbGV2ZWxfZnJhZ19zdGFydBgOIAMoDRIWCg5sZXZlbF9mcmFnX2VuZBgPIAMoDRIVCg1sZXZlbF9mcmFnX2F0GBAgAygNEhIKCmxldmVsX2tpbmQYESADKA0SEAoIbGV2ZWxfaXgYEiADKA0SFAoMbGV2ZWxfcGFyZW50GBMgAygFEigKBW1vZGVsGBQgASgLMhkudHVyYm8ucmVzdWx0LnYxLk1vZGVsUmVmEgsKA2RpbRgVIAEoDRIMCgRkYXRhGBYgASgMEg0KBWVtcHR5GBcgAygNIsABChZBbmFseXplUGlwZWxpbmVSZXF1ZXN0Eg4KBmRvY19pZBgBIAEoCRIMCgR0ZXh0GAIgASgJEhQKDHNvdXJjZV9ieXRlcxgDIAEoDBIsCgdvZmZzZXRzGAQgASgOMhsudHVyYm8ucmVzdWx0LnYxLk9mZnNldE1vZGUSDwoHcHJvZmlsZRgFIAEoCRINCgVzdGVwcxgGIAMoCRINCgVtb2RlbBgHIAEoCRIVCg1pbml0aWFsX2dyYXBoGAggASgMIp8CChdBbmFseXplUHJvZ3Jlc3NpdmVFdmVudBIQCghzZXF1ZW5jZRgBIAEoBBIzCgdzdGFydGVkGAIgASgLMiAudHVyYm8ucmVzdWx0LnYxLkFuYWx5c2lzU3RhcnRlZEgAEjsKDGxheWVyc19yZWFkeRgDIAEoCzIjLnR1cmJvLnJlc3VsdC52MS5BbmFseXNpc0xheWVyQmF0Y2hIABI7CgtzdGVwX2ZhaWxlZBgEIAEoCzIkLnR1cmJvLnJlc3VsdC52MS5BbmFseXNpc1N0ZXBGYWlsdXJlSAASOgoIY29tcGxldGUYBSABKAsyJi50dXJiby5yZXN1bHQudjEuRG9jdW1lbnRHcmFwaFJlc3BvbnNlSABCBwoFZXZlbnQiiwEKD0FuYWx5c2lzU3RhcnRlZBIOCgZkb2NfaWQYASABKAkSLAoHb2Zmc2V0cxgCIAEoDjIbLnR1cmJvLnJlc3VsdC52MS5PZmZzZXRNb2RlEhAKCHN0ZXBfaWRzGAMgAygJEigKBXZpZXdzGAQgAygLMhkudHVyYm8ucmVzdWx0LnYxLlZpZXdJbmZvIrUBChJBbmFseXNpc0xheWVyQmF0Y2gSFgoOY29tcGxldGVkX3N0ZXAYASABKAkSDQoFc3RhZ2UYAiABKA0SEwoLZHVyYXRpb25fdXMYAyABKAQSKgoGbGF5ZXJzGAQgAygLMhoudHVyYm8ucmVzdWx0LnYxLkxheWVySW5mbxIoCgV2aWV3cxgFIAMoCzIZLnR1cmJvLnJlc3VsdC52MS5WaWV3SW5mbxINCgVncmFwaBgGIAEoDCLEAQoTQW5hbHlzaXNTdGVwRmFpbHVyZRIPCgdzdGVwX2lkGAEgASgJEg0KBXN0YWdlGAIgASgNEhMKC2R1cmF0aW9uX3VzGAMgASgEEjAKB2ZhaWx1cmUYBCABKAsyHy50dXJiby5yZXN1bHQudjEuQW5hbHl6ZUZhaWx1cmUSGgoSc2tpcHBlZF9kZXBlbmRlbnRzGAUgAygJEioKBmxheWVycxgGIAMoCzIaLnR1cmJvLnJlc3VsdC52MS5MYXllckluZm8i5wEKFURvY3VtZW50V2luZG93UmVxdWVzdBINCgVncmFwaBgBIAEoDBIMCgR2aWV3GAIgASgNEiwKB29mZnNldHMYAyABKA4yGy50dXJiby5yZXN1bHQudjEuT2Zmc2V0TW9kZRIRCglwYWdlX3NpemUYBCABKA0SDAoEcGFnZRgFIAEoDRIVCg1maWx0ZXJfbGF5ZXJzGAYgAygJEhcKD2hlYXRtYXBfYnVja2V0cxgHIAEoDRIdChB4cmF5X3RhcmdldF92aWV3GAggASgNSACIAQFCEwoRX3hyYXlfdGFyZ2V0X3ZpZXci0gEKGURvY3VtZW50U2hhcGVTdW1tYXJ5UHJvdG8SDAoEdmlldxgBIAEoDRITCgt0b3RhbF9ieXRlcxgCIAEoDRITCgt0b3RhbF91bml0cxgDIAEoDRISCgp2aWV3X2NvdW50GAQgASgNEhMKC2xheWVyX2NvdW50GAUgASgNEioKBmxheWVycxgGIAMoCzIaLnR1cmJvLnJlc3VsdC52MS5MYXllckluZm8SKAoFdmlld3MYByADKAsyGS50dXJiby5yZXN1bHQudjEuVmlld0luZm8ipgEKE0RvY3VtZW50V2luZG93UHJvdG8SDAoEdmlldxgBIAEoDRISCgpieXRlX3N0YXJ0GAIgASgNEhAKCGJ5dGVfZW5kGAMgASgNEg0KBXN0YXJ0GAQgASgNEgsKA2VuZBgFIAEoDRIMCgRwYWdlGAYgASgNEhIKCnBhZ2VfY291bnQYByABKA0SDwoHY2xhbXBlZBgIIAEoCBIMCgR0ZXh0GAkgASgJImUKFlNlZ21lbnRBbm5vdGF0aW9uUHJvdG8SDQoFbGF5ZXIYASABKAkSEAoIaW5zdGFuY2UYAiABKAkSDAoEaXRlbRgDIAEoDRINCgVsYWJlbBgEIAEoCRINCgVzY29yZRgFIAEoASJ+ChRDb21iaW5lZFNlZ21lbnRQcm90bxINCgVzdGFydBgBIAEoDRILCgNlbmQYAiABKA0SDAoEdGV4dBgDIAEoCRI8Cgthbm5vdGF0aW9ucxgEIAMoCzInLnR1cmJvLnJlc3VsdC52MS5TZWdtZW50QW5ub3RhdGlvblByb3RvIlIKEURvY3VtZW50Q2hpcFByb3RvEg0KBWxheWVyGAEgASgJEhAKCGluc3RhbmNlGAIgASgJEg0KBWxhYmVsGAMgASgJEg0KBXNjb3JlGAQgASgBIqABChJIZWF0bWFwQnVja2V0UHJvdG8SDQoFc3RhcnQYASABKA0SCwoDZW5kGAIgASgNEg0KBXNjb3JlGAMgASgCEg0KBWNvdW50GAQgASgNEhkKDHByaW1hcnlfaXRlbRgFIAEoDUgAiAEBEhIKCmJ5dGVfc3RhcnQYBiABKA0SEAoIYnl0ZV9lbmQYByABKA1CDwoNX3ByaW1hcnlfaXRlbSKWAQoQSGVhdG1hcExhbmVQcm90bxIKCgJpZBgBIAEoCRIMCgRraW5kGAIgASgJEgwKBHZpZXcYAyABKA0SEQoJbWluX3Njb3JlGAQgASgCEhEKCW1heF9zY29yZRgFIAEoAhI0CgdidWNrZXRzGAYgAygLMiMudHVyYm8ucmVzdWx0LnYxLkhlYXRtYXBCdWNrZXRQcm90byK/AQoQWHJheUVkaXRSdW5Qcm90bxIUCgxwYXJlbnRfc3RhcnQYASABKA0SEgoKcGFyZW50X2VuZBgCIAEoDRITCgtjaGlsZF9zdGFydBgDIAEoDRIRCgljaGlsZF9lbmQYBCABKA0SMAoEa2luZBgFIAEoDjIiLnR1cmJvLnJlc3VsdC52MS5YcmF5RWRpdEtpbmRQcm90bxITCgtwYXJlbnRfdGV4dBgGIAEoCRISCgpjaGlsZF90ZXh0GAcgASgJItgBChRYcmF5U3RlcFN1bW1hcnlQcm90bxIMCgR2aWV3GAEgASgNEhgKC3BhcmVudF92aWV3GAIgASgNSACIAQESEAoIaW5zdGFuY2UYAyABKAkSEQoJYW5ub3RhdG9yGAQgASgJEg8KB3ZlcnNpb24YBSABKAkSEwoLY29uZmlnX2hhc2gYBiABKAQSEgoKZWRpdF9jb3VudBgHIAEoDRIUCgxwYXJlbnRfYnl0ZXMYCCABKA0SEwoLY2hpbGRfYnl0ZXMYCSABKA1CDgoMX3BhcmVudF92aWV3ItEBChZOb3JtYWxpemF0aW9uWHJheVByb3RvEhEKCWZyb21fdmlldxgBIAEoDRIPCgd0b192aWV3GAIgASgNEiwKB29mZnNldHMYAyABKA4yGy50dXJiby5yZXN1bHQudjEuT2Zmc2V0TW9kZRIvCgRydW5zGAQgAygLMiEudHVyYm8ucmVzdWx0LnYxLlhyYXlFZGl0UnVuUHJvdG8SNAoFc3RlcHMYBSADKAsyJS50dXJiby5yZXN1bHQudjEuWHJheVN0ZXBTdW1tYXJ5UHJvdG8i/wIKFkRvY3VtZW50V2luZG93UmVzcG9uc2USOwoHc3VtbWFyeRgBIAEoCzIqLnR1cmJvLnJlc3VsdC52MS5Eb2N1bWVudFNoYXBlU3VtbWFyeVByb3RvEjQKBndpbmRvdxgCIAEoCzIkLnR1cmJvLnJlc3VsdC52MS5Eb2N1bWVudFdpbmRvd1Byb3RvEjcKCHNlZ21lbnRzGAMgAygLMiUudHVyYm8ucmVzdWx0LnYxLkNvbWJpbmVkU2VnbWVudFByb3RvEjoKDmRvY3VtZW50X2NoaXBzGAQgAygLMiIudHVyYm8ucmVzdWx0LnYxLkRvY3VtZW50Q2hpcFByb3RvEjgKDWhlYXRtYXBfbGFuZXMYBSADKAsyIS50dXJiby5yZXN1bHQudjEuSGVhdG1hcExhbmVQcm90bxI6CgR4cmF5GAYgASgLMicudHVyYm8ucmVzdWx0LnYxLk5vcm1hbGl6YXRpb25YcmF5UHJvdG9IAIgBAUIHCgVfeHJheSJsCgpCdWRnZXRTcGVjEgwKBG5hbWUYASABKAkSEgoKbWF4X3Rva2VucxgCIAEoDRIVCg10YXJnZXRfdG9rZW5zGAMgASgNEhMKC3NpemVfd2VpZ2h0GAQgASgNEhAKCG92ZXJoZWFkGAUgASgNIvYBChRQcm9qZWN0Q2h1bmtzUmVxdWVzdBINCgVncmFwaBgBIAEoDBIsCgdvZmZzZXRzGAIgASgOMhsudHVyYm8ucmVzdWx0LnYxLk9mZnNldE1vZGUSEAoIcGNhX2RpbXMYAyABKA0SGgoSbWF4X3N0YWNrX3NlZ21lbnRzGAQgASgNEhsKE2luY2x1ZGVfZHJpZnRfY3VydmUYBSABKAgSHwoXaW5jbHVkZV9zdHJhdGVneV9tYXRyaXgYBiABKAgSNQoQc3RyYXRlZ3lfYnVkZ2V0cxgHIAMoCzIbLnR1cmJvLnJlc3VsdC52MS5CdWRnZXRTcGVjIrABChNQcm9qZWN0ZWRQb2ludFByb3RvEgwKBGl0ZW0YASABKA0SDQoFc3RhcnQYAiABKA0SCwoDZW5kGAMgASgNEg0KBWxldmVsGAQgASgNEgkKAXgYBSABKAISCQoBeRgGIAEoAhIJCgF6GAcgASgCEg0KBWVtcHR5GAggASgIEhwKD2FkamFjZW50X2Nvc2luZRgJIAEoAkgAiAEBQhIKEF9hZGphY2VudF9jb3NpbmUipQEKFVZlY3RvclByb2plY3Rpb25Qcm90bxIoCgVtb2RlbBgBIAEoCzIZLnR1cmJvLnJlc3VsdC52MS5Nb2RlbFJlZhIQCghwY2FfZGltcxgCIAEoDRIaChJleHBsYWluZWRfdmFyaWFuY2UYAyADKAISNAoGcG9pbnRzGAQgAygLMiQudHVyYm8ucmVzdWx0LnYxLlByb2plY3RlZFBvaW50UHJvdG8i+wEKEkRyaWZ0R2FwUG9pbnRQcm90bxIQCghzZW50ZW5jZRgBIAEoDRISCgpnYXBfb2Zmc2V0GAIgASgNEh0KEHJhd19jb3NpbmVfbWljcm8YAyABKANIAIgBARIiChVzbW9vdGhlZF9jb3NpbmVfbWljcm8YBCABKANIAYgBARIZChFkZXB0aF9zY29yZV9taWNybxgFIAEoAxIMCgRjb3N0GAYgASgFEg4KBnJlYXNvbhgHIAEoDRIUCgxpc19jaHVua19jdXQYCCABKAhCEwoRX3Jhd19jb3NpbmVfbWljcm9CGAoWX3Ntb290aGVkX2Nvc2luZV9taWNybyJvChdCb3VuZGFyeURyaWZ0Q3VydmVQcm90bxIOCgZ3aW5kb3cYASABKA0SEQoJc21vb3RoaW5nGAIgASgNEjEKBGdhcHMYAyADKAsyIy50dXJiby5yZXN1bHQudjEuRHJpZnRHYXBQb2ludFByb3RvIqIBChZUZXJtV2VpZ2h0U2VnbWVudFByb3RvEgwKBHRlcm0YASABKAkSDwoHdGVybV9pZBgCIAEoDRIRCglmcmVxdWVuY3kYAyABKA0SDgoGd2VpZ2h0GAQgASgCEg0KBXNoYXJlGAUgASgCEhMKC2lzX2V4cGFuZGVkGAYgASgIEhEKCW9jY19zdGFydBgHIAMoDRIPCgdvY2NfZW5kGAggAygNIuwBChRUZXJtVmVjdG9yU3RhY2tQcm90bxIMCgR1bml0GAEgASgNEjkKCHNlZ21lbnRzGAIgAygLMicudHVyYm8ucmVzdWx0LnYxLlRlcm1XZWlnaHRTZWdtZW50UHJvdG8SEwoLdG90YWxfdGVybXMYAyABKA0SFwoPdG90YWxfZnJlcXVlbmN5GAQgASgEEhQKDHRvdGFsX3dlaWdodBgFIAEoAhIYChBvdGhlcl90ZXJtX2NvdW50GAYgASgNEhcKD290aGVyX2ZyZXF1ZW5jeRgHIAEoBBIUCgxvdGhlcl93ZWlnaHQYCCABKAIi+gEKHFN0cmF0ZWd5Q29tcGFyaXNvbkVudHJ5UHJvdG8SDAoEbmFtZRgBIAEoCRIrCgZidWRnZXQYAiABKAsyGy50dXJiby5yZXN1bHQudjEuQnVkZ2V0U3BlYxITCgtjaHVua19jb3VudBgDIAEoDRIZChFvdmVyX2J1ZGdldF9jb3VudBgEIAEoDRISCgp0b3RhbF9jb3N0GAUgASgDEiIKGm1hdGNoZWRfY3V0c193aXRoX2Jhc2VsaW5lGAYgASgNEiMKG2Rpc2FncmVlbWVudHNfd2l0aF9iYXNlbGluZRgHIAEoDRISCgpjdXRfc3RhcnRzGAggAygNIlgKE1N0cmF0ZWd5TWF0cml4UHJvdG8SQQoKc3RyYXRlZ2llcxgBIAMoCzItLnR1cmJvLnJlc3VsdC52MS5TdHJhdGVneUNvbXBhcmlzb25FbnRyeVByb3RvIqIDChVQcm9qZWN0Q2h1bmtzUmVzcG9uc2USOgoKcHJvamVjdGlvbhgBIAEoCzImLnR1cmJvLnJlc3VsdC52MS5WZWN0b3JQcm9qZWN0aW9uUHJvdG8SQgoLZHJpZnRfY3VydmUYAiABKAsyKC50dXJiby5yZXN1bHQudjEuQm91bmRhcnlEcmlmdEN1cnZlUHJvdG9IAIgBARJAChFjaHVua190ZXJtX3N0YWNrcxgDIAMoCzIlLnR1cmJvLnJlc3VsdC52MS5UZXJtVmVjdG9yU3RhY2tQcm90bxJHChNkb2N1bWVudF90ZXJtX3N0YWNrGAQgASgLMiUudHVyYm8ucmVzdWx0LnYxLlRlcm1WZWN0b3JTdGFja1Byb3RvSAGIAQESQgoPc3RyYXRlZ3lfbWF0cml4GAUgASgLMiQudHVyYm8ucmVzdWx0LnYxLlN0cmF0ZWd5TWF0cml4UHJvdG9IAogBAUIOCgxfZHJpZnRfY3VydmVCFgoUX2RvY3VtZW50X3Rlcm1fc3RhY2tCEgoQX3N0cmF0ZWd5X21hdHJpeCIaChhMaXN0T3V0cHV0Rm9ybWF0c1JlcXVlc3QiwgEKG091dHB1dEZvcm1hdERlc2NyaXB0b3JQcm90bxIRCglmb3JtYXRfaWQYASABKAkSFAoMZGlzcGxheV9uYW1lGAIgASgJEhIKCm1lZGlhX3R5cGUYAyABKAkSFgoOZmlsZV9leHRlbnNpb24YBCABKAkSDgoGYmluYXJ5GAUgASgIEhAKCGxvc3NsZXNzGAYgASgIEhcKD3JlcXVpcmVkX2xheWVycxgHIAMoCRITCgtkZXNjcmlwdGlvbhgIIAEoCSJaChlMaXN0T3V0cHV0Rm9ybWF0c1Jlc3BvbnNlEj0KB2Zvcm1hdHMYASADKAsyLC50dXJiby5yZXN1bHQudjEuT3V0cHV0Rm9ybWF0RGVzY3JpcHRvclByb3RvIncKFUZvcm1hdERvY3VtZW50UmVxdWVzdBIOCgZkb2NfaWQYASABKAkSDQoFZ3JhcGgYAiABKAwSEQoJZm9ybWF0X2lkGAMgASgJEiwKB29mZnNldHMYBCABKA4yGy50dXJiby5yZXN1bHQudjEuT2Zmc2V0TW9kZSJoChZGb3JtYXREb2N1bWVudFJlc3BvbnNlEhEKCWZvcm1hdF9pZBgBIAEoCRISCgptZWRpYV90eXBlGAIgASgJEhYKDmZpbGVfZXh0ZW5zaW9uGAMgASgJEg8KB2NvbnRlbnQYBCABKAwqRQoMQW5hbHl6ZU9yZGVyEhcKE0FOQUxZWkVfT1JERVJfSU5QVVQQABIcChhBTkFMWVpFX09SREVSX0NPTVBMRVRJT04QASpuCg1BbmFseXplT3V0cHV0EhoKFkFOQUxZWkVfT1VUUFVUX1ZFQ1RPUlMQABIfChtBTkFMWVpFX09VVFBVVF9TUEFSU0VfVEVSTVMQARIgChxBTkFMWVpFX09VVFBVVF9UT0tFTl9WRUNUT1JTEAIqmwEKEVhyYXlFZGl0S2luZFByb3RvEh4KGlhSQVlfRURJVF9LSU5EX1BST1RPX0VRVUFMEAASIQodWFJBWV9FRElUX0tJTkRfUFJPVE9fUkVQTEFDRUQQARIhCh1YUkFZX0VESVRfS0lORF9QUk9UT19JTlNFUlRFRBACEiAKHFhSQVlfRURJVF9LSU5EX1BST1RPX0RFTEVURUQQAzKoDgoPQW5hbHlzaXNTZXJ2aWNlEl0KDENhcGFiaWxpdGllcxIkLnR1cmJvLnJlc3VsdC52MS5DYXBhYmlsaXRpZXNSZXF1ZXN0GiUudHVyYm8ucmVzdWx0LnYxLkNhcGFiaWxpdGllc1Jlc3BvbnNlIgASVwoKTG9hZEJ1bmRsZRIiLnR1cmJvLnJlc3VsdC52MS5Mb2FkQnVuZGxlUmVxdWVzdBojLnR1cmJvLnJlc3VsdC52MS5Mb2FkQnVuZGxlUmVzcG9uc2UiABJdCgxVbmxvYWRCdW5kbGUSJC50dXJiby5yZXN1bHQudjEuVW5sb2FkQnVuZGxlUmVxdWVzdBolLnR1cmJvLnJlc3VsdC52MS5VbmxvYWRCdW5kbGVSZXNwb25zZSIAEmAKFFJvdW5kVHJpcFNwYXJzZVRlcm1zEiIudHVyYm8ucmVzdWx0LnYxLlNwYXJzZVRlcm1zUmVzdWx0GiIudHVyYm8ucmVzdWx0LnYxLlNwYXJzZVRlcm1zUmVzdWx0IgASYwoVUm91bmRUcmlwVG9rZW5WZWN0b3JzEiMudHVyYm8ucmVzdWx0LnYxLlRva2VuVmVjdG9yc1Jlc3VsdBojLnR1cmJvLnJlc3VsdC52MS5Ub2tlblZlY3RvcnNSZXN1bHQiABJjChVSb3VuZFRyaXBSZXJhbmtTY29yZXMSIy50dXJiby5yZXN1bHQudjEuUmVyYW5rU2NvcmVzUmVzdWx0GiMudHVyYm8ucmVzdWx0LnYxLlJlcmFua1Njb3Jlc1Jlc3VsdCIAEl0KE1JvdW5kVHJpcEV4dHJhY3Rpb24SIS50dXJiby5yZXN1bHQudjEuRXh0cmFjdGlvblJlc3VsdBohLnR1cmJvLnJlc3VsdC52MS5FeHRyYWN0aW9uUmVzdWx0IgASVAoQUm91bmRUcmlwUmVjb3JkcxIeLnR1cmJvLnJlc3VsdC52MS5SZWNvcmRzUmVzdWx0Gh4udHVyYm8ucmVzdWx0LnYxLlJlY29yZHNSZXN1bHQiABJgChRSb3VuZFRyaXBDaHVua0xldmVscxIiLnR1cmJvLnJlc3VsdC52MS5DaHVua0xldmVsc1Jlc3VsdBoiLnR1cmJvLnJlc3VsdC52MS5DaHVua0xldmVsc1Jlc3VsdCIAEmkKFlJvdW5kVHJpcERvY3VtZW50R3JhcGgSJS50dXJiby5yZXN1bHQudjEuRG9jdW1lbnRHcmFwaFJlcXVlc3QaJi50dXJiby5yZXN1bHQudjEuRG9jdW1lbnRHcmFwaFJlc3BvbnNlIgASbAoUSW5zcGVjdERvY3VtZW50R3JhcGgSLC50dXJiby5yZXN1bHQudjEuSW5zcGVjdERvY3VtZW50R3JhcGhSZXF1ZXN0GiQudHVyYm8ucmVzdWx0LnYxLlR5cGVkRG9jdW1lbnRSZXN1bHQiABJmChZSb3VuZFRyaXBUeXBlZERvY3VtZW50EiQudHVyYm8ucmVzdWx0LnYxLlR5cGVkRG9jdW1lbnRSZXN1bHQaJC50dXJiby5yZXN1bHQudjEuVHlwZWREb2N1bWVudFJlc3VsdCIAEmQKDUFuYWx5emVTdHJlYW0SJS50dXJiby5yZXN1bHQudjEuQW5hbHl6ZVN0cmVhbVJlcXVlc3QaJi50dXJiby5yZXN1bHQudjEuQW5hbHl6ZVN0cmVhbVJlc3BvbnNlIgAoATABEnMKGkFuYWx5emVEb2N1bWVudFByb2dyZXNzaXZlEicudHVyYm8ucmVzdWx0LnYxLkFuYWx5emVQaXBlbGluZVJlcXVlc3QaKC50dXJiby5yZXN1bHQudjEuQW5hbHl6ZVByb2dyZXNzaXZlRXZlbnQiADABEmoKFUluc3BlY3REb2N1bWVudFdpbmRvdxImLnR1cmJvLnJlc3VsdC52MS5Eb2N1bWVudFdpbmRvd1JlcXVlc3QaJy50dXJiby5yZXN1bHQudjEuRG9jdW1lbnRXaW5kb3dSZXNwb25zZSIAEmAKDVByb2plY3RDaHVua3MSJS50dXJiby5yZXN1bHQudjEuUHJvamVjdENodW5rc1JlcXVlc3QaJi50dXJiby5yZXN1bHQudjEuUHJvamVjdENodW5rc1Jlc3BvbnNlIgASbAoRTGlzdE91dHB1dEZvcm1hdHMSKS50dXJiby5yZXN1bHQudjEuTGlzdE91dHB1dEZvcm1hdHNSZXF1ZXN0GioudHVyYm8ucmVzdWx0LnYxLkxpc3RPdXRwdXRGb3JtYXRzUmVzcG9uc2UiABJjCg5Gb3JtYXREb2N1bWVudBImLnR1cmJvLnJlc3VsdC52MS5Gb3JtYXREb2N1bWVudFJlcXVlc3QaJy50dXJiby5yZXN1bHQudjEuRm9ybWF0RG9jdW1lbnRSZXNwb25zZSIAUABiBnByb3RvMw", [file_turbo_typed_annotations]);
var AnalysisService = /* @__PURE__ */ serviceDesc(file_turbo_analysis, 0);

// demo/web/src/gen/turbo_search_pb.ts
var file_turbo_search = /* @__PURE__ */ fileDesc("ChJ0dXJib19zZWFyY2gucHJvdG8SD3R1cmJvLnNlYXJjaC52MSI9Cg1Nb2RlbElkZW50aXR5EgoKAmlkGAEgASgJEhAKCHJldmlzaW9uGAIgASgJEg4KBnNoYTI1NhgDIAEoCSKCAQoJU3BhY2VTcGVjEi0KBW1vZGVsGAEgASgLMh4udHVyYm8uc2VhcmNoLnYxLk1vZGVsSWRlbnRpdHkSCwoDZGltGAIgASgNEjkKDW5vcm1hbGl6YXRpb24YAyABKA4yIi50dXJiby5zZWFyY2gudjEuTm9ybWFsaXphdGlvbk1vZGUiqQIKCUluZGV4U3BlYxISCgppbmRleF9uYW1lGAEgASgJEikKBXNwYWNlGAIgASgLMhoudHVyYm8uc2VhcmNoLnYxLlNwYWNlU3BlYxIvCgZtZXRyaWMYAyABKA4yHy50dXJiby5zZWFyY2gudjEuRGlzdGFuY2VNZXRyaWMSMwoHc3RvcmFnZRgEIAEoDjIiLnR1cmJvLnNlYXJjaC52MS5WZWN0b3JTdG9yYWdlS2luZBISCgpxdWFudF9iaXRzGAUgASgNEhcKD3F1YW50X2NhbGlicmF0ZRgGIAEoCBIXCg9jb2xsZWN0aW9uX25hbWUYByABKAkSFAoMZW1iZWRfYnVuZGxlGAggASgJEhsKE2NodW5rX2J1ZGdldF90b2tlbnMYCSABKA0i7wIKDEluZGV4U3VtbWFyeRISCgppbmRleF9uYW1lGAEgASgJEikKBXNwYWNlGAIgASgLMhoudHVyYm8uc2VhcmNoLnYxLlNwYWNlU3BlYxIvCgZtZXRyaWMYAyABKA4yHy50dXJiby5zZWFyY2gudjEuRGlzdGFuY2VNZXRyaWMSMwoHc3RvcmFnZRgEIAEoDjIiLnR1cmJvLnNlYXJjaC52MS5WZWN0b3JTdG9yYWdlS2luZBISCgpxdWFudF9iaXRzGAUgASgNEg4KBnNlYWxlZBgGIAEoCBIWCg5kb2N1bWVudF9jb3VudBgHIAEoDRITCgtjaHVua19jb3VudBgIIAEoDRIXCg9kZW5zZV9yb3dfY291bnQYCSABKA0SGAoQc3BhcnNlX3Jvd19jb3VudBgKIAEoDRIdChV0b2tlbl9jYW5kaWRhdGVfY291bnQYCyABKA0SFwoPY29sbGVjdGlvbl9uYW1lGAwgASgJIkkKGFNlYXJjaFByb3ZpZGVyRGVzY3JpcHRvchIKCgJpZBgBIAEoCRIMCgRraW5kGAIgASgJEhMKC2Rlc2NyaXB0aW9uGAMgASgJIhwKGkxpc3RTZWFyY2hQcm92aWRlcnNSZXF1ZXN0IlsKG0xpc3RTZWFyY2hQcm92aWRlcnNSZXNwb25zZRI8Cglwcm92aWRlcnMYASADKAsyKS50dXJiby5zZWFyY2gudjEuU2VhcmNoUHJvdmlkZXJEZXNjcmlwdG9yIhQKEkxpc3RJbmRleGVzUmVxdWVzdCJFChNMaXN0SW5kZXhlc1Jlc3BvbnNlEi4KB2luZGV4ZXMYASADKAsyHS50dXJiby5zZWFyY2gudjEuSW5kZXhTdW1tYXJ5IlYKEkNyZWF0ZUluZGV4UmVxdWVzdBIoCgRzcGVjGAEgASgLMhoudHVyYm8uc2VhcmNoLnYxLkluZGV4U3BlYxIWCg5sb2FkX2Zyb21fcGF0aBgCIAEoCSJFChNDcmVhdGVJbmRleFJlc3BvbnNlEi4KB3N1bW1hcnkYASABKAsyHS50dXJiby5zZWFyY2gudjEuSW5kZXhTdW1tYXJ5IigKEkRlbGV0ZUluZGV4UmVxdWVzdBISCgppbmRleF9uYW1lGAEgASgJIhUKE0RlbGV0ZUluZGV4UmVzcG9uc2UifgoQU3BhcnNlRW50cnlQcm90bxIPCgd0ZXJtX2lkGAEgASgNEhEKCXRlcm1fdGV4dBgCIAEoCRIOCgZ3ZWlnaHQYAyABKAISEgoKc3Bhbl9zdGFydBgEIAEoDRIQCghzcGFuX2VuZBgFIAEoDRIQCghoYXNfc3BhbhgGIAEoCCJaChBUb2tlblZlY3RvcklucHV0Eg4KBnZhbHVlcxgBIAMoAhISCgpzcGFuX3N0YXJ0GAIgASgNEhAKCHNwYW5fZW5kGAMgASgNEhAKCGhhc19zcGFuGAQgASgIIpMDCgpDaHVua0lucHV0EhAKCGNodW5rX2lkGAEgASgJEhIKCmxldmVsX2tpbmQYAiABKAkSEgoKc3Bhbl9zdGFydBgDIAEoDRIQCghzcGFuX2VuZBgEIAEoDRIUCgxwYXJlbnRfc3RhcnQYBSABKA0SEgoKcGFyZW50X2VuZBgGIAEoDRISCgpoYXNfcGFyZW50GAcgASgIEhQKDGRlbnNlX3ZlY3RvchgIIAMoAhI3CgxzcGFyc2VfdGVybXMYCSADKAsyIS50dXJiby5zZWFyY2gudjEuU3BhcnNlRW50cnlQcm90bxI4Cg10b2tlbl92ZWN0b3JzGAogAygLMiEudHVyYm8uc2VhcmNoLnYxLlRva2VuVmVjdG9ySW5wdXQSPwoKYXR0cmlidXRlcxgLIAMoCzIrLnR1cmJvLnNlYXJjaC52MS5DaHVua0lucHV0LkF0dHJpYnV0ZXNFbnRyeRoxCg9BdHRyaWJ1dGVzRW50cnkSCwoDa2V5GAEgASgJEg0KBXZhbHVlGAIgASgJOgI4ASKCAgoUSW5kZXhEb2N1bWVudFJlcXVlc3QSEgoKaW5kZXhfbmFtZRgBIAEoCRIOCgZkb2NfaWQYAiABKAkSDAoEdGV4dBgDIAEoCRINCgVncmFwaBgEIAEoDBIrCgZjaHVua3MYBSADKAsyGy50dXJiby5zZWFyY2gudjEuQ2h1bmtJbnB1dBJJCgphdHRyaWJ1dGVzGAYgAygLMjUudHVyYm8uc2VhcmNoLnYxLkluZGV4RG9jdW1lbnRSZXF1ZXN0LkF0dHJpYnV0ZXNFbnRyeRoxCg9BdHRyaWJ1dGVzRW50cnkSCwoDa2V5GAEgASgJEg0KBXZhbHVlGAIgASgJOgI4ASKWAQoWSW5kZXhEb2N1bWVudHNSZXNwb25zZRISCgppbmRleF9uYW1lGAEgASgJEhkKEWluZGV4ZWRfZG9jdW1lbnRzGAIgASgNEhYKDmluZGV4ZWRfY2h1bmtzGAMgASgNEjUKBWRyaWZ0GAQgASgLMiYudHVyYm8uc2VhcmNoLnYxLlZvY2FidWxhcnlEcmlmdFJlcG9ydCJ6Cg5TZW1hbnRpY0NsYXVzZRIUCgxxdWVyeV92ZWN0b3IYASADKAISLwoLcXVlcnlfc3BhY2UYAiABKAsyGi50dXJiby5zZWFyY2gudjEuU3BhY2VTcGVjEhIKCnF1ZXJ5X3RleHQYAyABKAkSDQoFdG9wX2sYBCABKA0iWAoKVGVybUNsYXVzZRINCgV0ZXJtcxgBIAMoCRIsCgRtb2RlGAIgASgOMh4udHVyYm8uc2VhcmNoLnYxLlRlcm1NYXRjaE1vZGUSDQoFdG9wX2sYAyABKA0iOgoMUGhyYXNlQ2xhdXNlEg0KBXRlcm1zGAEgAygJEgwKBHNsb3AYAiABKA0SDQoFdG9wX2sYAyABKA0iVQoMU3BhcnNlQ2xhdXNlEjYKC3F1ZXJ5X3Rlcm1zGAEgAygLMiEudHVyYm8uc2VhcmNoLnYxLlNwYXJzZUVudHJ5UHJvdG8SDQoFdG9wX2sYAiABKA0idAoVTGF0ZUludGVyYWN0aW9uQ2xhdXNlEhsKE3F1ZXJ5X3Rva2VuX3ZlY3RvcnMYASADKAISLwoLcXVlcnlfc3BhY2UYAiABKAsyGi50dXJiby5zZWFyY2gudjEuU3BhY2VTcGVjEg0KBXRvcF9rGAMgASgNIr8BCgxGaWx0ZXJDbGF1c2USDgoGZG9jX2lkGAEgASgJEhIKCmxldmVsX2tpbmQYAiABKAkSQQoKYXR0cmlidXRlcxgDIAMoCzItLnR1cmJvLnNlYXJjaC52MS5GaWx0ZXJDbGF1c2UuQXR0cmlidXRlc0VudHJ5EhUKDWNvbnRhaW5zX3RleHQYBCABKAkaMQoPQXR0cmlidXRlc0VudHJ5EgsKA2tleRgBIAEoCRINCgV2YWx1ZRgCIAEoCToCOAEiRwoLQm9vc3RDbGF1c2USKQoFY2hpbGQYASABKAsyGi50dXJiby5zZWFyY2gudjEuUXVlcnlOb2RlEg0KBWJvb3N0GAIgASgCIqUBCgpKb2luQ2xhdXNlEikKAm9wGAEgASgOMh0udHVyYm8uc2VhcmNoLnYxLkpvaW5PcGVyYXRvchIvCgZmdXNpb24YAiABKA4yHy50dXJiby5zZWFyY2gudjEuRnVzaW9uU3RyYXRlZ3kSLAoIY2hpbGRyZW4YAyADKAsyGi50dXJiby5zZWFyY2gudjEuUXVlcnlOb2RlEg0KBXJyZl9rGAQgASgCIqoDCglRdWVyeU5vZGUSMwoIc2VtYW50aWMYASABKAsyHy50dXJiby5zZWFyY2gudjEuU2VtYW50aWNDbGF1c2VIABIrCgR0ZXJtGAIgASgLMhsudHVyYm8uc2VhcmNoLnYxLlRlcm1DbGF1c2VIABIvCgZwaHJhc2UYAyABKAsyHS50dXJiby5zZWFyY2gudjEuUGhyYXNlQ2xhdXNlSAASLwoGc3BhcnNlGAQgASgLMh0udHVyYm8uc2VhcmNoLnYxLlNwYXJzZUNsYXVzZUgAEkIKEGxhdGVfaW50ZXJhY3Rpb24YBSABKAsyJi50dXJiby5zZWFyY2gudjEuTGF0ZUludGVyYWN0aW9uQ2xhdXNlSAASLwoGZmlsdGVyGAYgASgLMh0udHVyYm8uc2VhcmNoLnYxLkZpbHRlckNsYXVzZUgAEi0KBWJvb3N0GAcgASgLMhwudHVyYm8uc2VhcmNoLnYxLkJvb3N0Q2xhdXNlSAASKwoEam9pbhgIIAEoCzIbLnR1cmJvLnNlYXJjaC52MS5Kb2luQ2xhdXNlSABCCAoGY2xhdXNlIkEKE1BhcmVudEV4cGFuc2lvblNwZWMSDwoHZW5hYmxlZBgBIAEoCBIZChFtYXhfcGFyZW50X3Rva2VucxgCIAEoDSKWAgoNU2VhcmNoUmVxdWVzdBISCgppbmRleF9uYW1lGAEgASgJEhUKDW5hdHVyYWxfcXVlcnkYAiABKAkSLQoJcXVlcnlfYXN0GAMgASgLMhoudHVyYm8uc2VhcmNoLnYxLlF1ZXJ5Tm9kZRIUCgxxdWVyeV92ZWN0b3IYBCADKAISLwoLcXVlcnlfc3BhY2UYBSABKAsyGi50dXJiby5zZWFyY2gudjEuU3BhY2VTcGVjEg0KBXRvcF9rGAYgASgNEj4KEHBhcmVudF9leHBhbnNpb24YByABKAsyJC50dXJiby5zZWFyY2gudjEuUGFyZW50RXhwYW5zaW9uU3BlYxIVCg1yZXJhbmtfYnVuZGxlGAggASgJIicKCVNwYW5Qcm90bxINCgVzdGFydBgBIAEoDRILCgNlbmQYAiABKA0iZQoLTWF0Y2hlZFNwYW4SDgoGc291cmNlGAEgASgJEg0KBWxhYmVsGAIgASgJEigKBHNwYW4YAyABKAsyGi50dXJiby5zZWFyY2gudjEuU3BhblByb3RvEg0KBXNjb3JlGAQgASgCIsgBChBDaHVua0F0dHJpYnV0aW9uEjMKDW1hdGNoZWRfc3BhbnMYASADKAsyHC50dXJiby5zZWFyY2gudjEuTWF0Y2hlZFNwYW4SSgoNc2lnbmFsX3Njb3JlcxgCIAMoCzIzLnR1cmJvLnNlYXJjaC52MS5DaHVua0F0dHJpYnV0aW9uLlNpZ25hbFNjb3Jlc0VudHJ5GjMKEVNpZ25hbFNjb3Jlc0VudHJ5EgsKA2tleRgBIAEoCRINCgV2YWx1ZRgCIAEoAjoCOAEikwMKCVNlYXJjaEhpdBIQCghjaHVua19pZBgBIAEoCRIOCgZkb2NfaWQYAiABKAkSEgoKbGV2ZWxfa2luZBgDIAEoCRINCgVzY29yZRgEIAEoAhIuCgpjaHVua19zcGFuGAUgASgLMhoudHVyYm8uc2VhcmNoLnYxLlNwYW5Qcm90bxIvCgtwYXJlbnRfc3BhbhgGIAEoCzIaLnR1cmJvLnNlYXJjaC52MS5TcGFuUHJvdG8SEgoKaGFzX3BhcmVudBgHIAEoCBIMCgR0ZXh0GAggASgJEhMKC3BhcmVudF90ZXh0GAkgASgJEjYKC2F0dHJpYnV0aW9uGAogASgLMiEudHVyYm8uc2VhcmNoLnYxLkNodW5rQXR0cmlidXRpb24SPgoKYXR0cmlidXRlcxgLIAMoCzIqLnR1cmJvLnNlYXJjaC52MS5TZWFyY2hIaXQuQXR0cmlidXRlc0VudHJ5GjEKD0F0dHJpYnV0ZXNFbnRyeRILCgNrZXkYASABKAkSDQoFdmFsdWUYAiABKAk6AjgBIlgKC0hlYXRTZWdtZW50EigKBHNwYW4YASABKAsyGi50dXJiby5zZWFyY2gudjEuU3BhblByb3RvEg0KBXNjb3JlGAIgASgCEhAKCGNodW5rX2lkGAMgASgJImEKDERvY3VtZW50SGVhdBIOCgZkb2NfaWQYASABKAkSEQoJbWF4X3Njb3JlGAIgASgCEi4KCHNlZ21lbnRzGAMgAygLMhwudHVyYm8uc2VhcmNoLnYxLkhlYXRTZWdtZW50Il8KFEV4dHJhY3RlZEZpbHRlclByb3RvEgwKBGtpbmQYASABKAkSDwoHc3VtbWFyeRgCIAEoCRIoCgRzcGFuGAMgASgLMhoudHVyYm8uc2VhcmNoLnYxLlNwYW5Qcm90byLoAQoOU2VhcmNoUmVzcG9uc2USFgoOcmVzb2x2ZWRfaW5kZXgYASABKAkSKAoEaGl0cxgCIAMoCzIaLnR1cmJvLnNlYXJjaC52MS5TZWFyY2hIaXQSNAoNZG9jdW1lbnRfaGVhdBgDIAMoCzIdLnR1cmJvLnNlYXJjaC52MS5Eb2N1bWVudEhlYXQSHAoUYW5hbHl6ZWRfcXVlcnlfdGVybXMYBCADKAkSQAoRZXh0cmFjdGVkX2ZpbHRlcnMYBSADKAsyJS50dXJiby5zZWFyY2gudjEuRXh0cmFjdGVkRmlsdGVyUHJvdG8iNwoTUGVyc2lzdEluZGV4UmVxdWVzdBISCgppbmRleF9uYW1lGAEgASgJEgwKBHBhdGgYAiABKAkiQQoUUGVyc2lzdEluZGV4UmVzcG9uc2USEgoKaW5kZXhfbmFtZRgBIAEoCRIVCg1ieXRlc193cml0dGVuGAIgASgEIiYKEFNlYWxJbmRleFJlcXVlc3QSEgoKaW5kZXhfbmFtZRgBIAEoCSJDChFTZWFsSW5kZXhSZXNwb25zZRIuCgdzdW1tYXJ5GAEgASgLMh0udHVyYm8uc2VhcmNoLnYxLkluZGV4U3VtbWFyeSJrCg5SZWluZGV4UmVxdWVzdBIUCgxzb3VyY2VfaW5kZXgYASABKAkSLwoLdGFyZ2V0X3NwZWMYAiABKAsyGi50dXJiby5zZWFyY2gudjEuSW5kZXhTcGVjEhIKCnN3YXBfYWxpYXMYAyABKAkiWAoPUmVpbmRleFJlc3BvbnNlEi4KB3N1bW1hcnkYASABKAsyHS50dXJiby5zZWFyY2gudjEuSW5kZXhTdW1tYXJ5EhUKDXN3YXBwZWRfYWxpYXMYAiABKAkiNAoPU2V0QWxpYXNSZXF1ZXN0Eg0KBWFsaWFzGAEgASgJEhIKCmluZGV4X25hbWUYAiABKAkiEgoQU2V0QWxpYXNSZXNwb25zZSIjChJEZWxldGVBbGlhc1JlcXVlc3QSDQoFYWxpYXMYASABKAkiFQoTRGVsZXRlQWxpYXNSZXNwb25zZSIvCgpBbGlhc0VudHJ5Eg0KBWFsaWFzGAEgASgJEhIKCmluZGV4X25hbWUYAiABKAkiFAoSTGlzdEFsaWFzZXNSZXF1ZXN0IkMKE0xpc3RBbGlhc2VzUmVzcG9uc2USLAoHYWxpYXNlcxgBIAMoCzIbLnR1cmJvLnNlYXJjaC52MS5BbGlhc0VudHJ5Ij4KElRlcm1GcmVxdWVuY3lFbnRyeRIMCgR0ZXJtGAEgASgJEhoKEmRvY3VtZW50X2ZyZXF1ZW5jeRgCIAEoBCJkChRTZXRDb2xsZWN0aW9uUmVxdWVzdBIXCg9jb2xsZWN0aW9uX25hbWUYASABKAkSFgoOYmFzZWxpbmVfdGVybXMYAiADKAkSGwoTb292X2FsZXJ0X3RocmVzaG9sZBgDIAEoASJMChVTZXRDb2xsZWN0aW9uUmVzcG9uc2USMwoKY29sbGVjdGlvbhgBIAEoCzIfLnR1cmJvLnNlYXJjaC52MS5Db2xsZWN0aW9uSW5mbyIvChRHZXRDb2xsZWN0aW9uUmVxdWVzdBIXCg9jb2xsZWN0aW9uX25hbWUYASABKAkiTAoVR2V0Q29sbGVjdGlvblJlc3BvbnNlEjMKCmNvbGxlY3Rpb24YASABKAsyHy50dXJiby5zZWFyY2gudjEuQ29sbGVjdGlvbkluZm8iGAoWTGlzdENvbGxlY3Rpb25zUmVxdWVzdCJPChdMaXN0Q29sbGVjdGlvbnNSZXNwb25zZRI0Cgtjb2xsZWN0aW9ucxgBIAMoCzIfLnR1cmJvLnNlYXJjaC52MS5Db2xsZWN0aW9uSW5mbyIyChdEZWxldGVDb2xsZWN0aW9uUmVxdWVzdBIXCg9jb2xsZWN0aW9uX25hbWUYASABKAkiGgoYRGVsZXRlQ29sbGVjdGlvblJlc3BvbnNlIsEBCg5Db2xsZWN0aW9uSW5mbxIXCg9jb2xsZWN0aW9uX25hbWUYASABKAkSEwoLdG90YWxfdW5pdHMYAiABKAQSEwoLbWVhbl9sZW5ndGgYAyABKAESFwoPdm9jYWJ1bGFyeV9zaXplGAQgASgEEhsKE29vdl9hbGVydF90aHJlc2hvbGQYBSABKAESNgoJdG9wX3Rlcm1zGAYgAygLMiMudHVyYm8uc2VhcmNoLnYxLlRlcm1GcmVxdWVuY3lFbnRyeSLAAQoVVm9jYWJ1bGFyeURyaWZ0UmVwb3J0EhcKD2NvbGxlY3Rpb25fbmFtZRgBIAEoCRIbChN2b2NhYnVsYXJ5X2NvdmVyYWdlGAIgASgBEhEKCW9vdl9yYXRpbxgDIAEoARIRCgluZXdfdGVybXMYBCADKAkSNgoJdG9wX3Rlcm1zGAUgAygLMiMudHVyYm8uc2VhcmNoLnYxLlRlcm1GcmVxdWVuY3lFbnRyeRITCgt0b3RhbF91bml0cxgGIAEoBCJIChZXYXRjaENvbGxlY3Rpb25SZXF1ZXN0EhcKD2NvbGxlY3Rpb25fbmFtZRgBIAEoCRIVCg1taW5fb292X3JhdGlvGAIgASgBIngKD0NvbGxlY3Rpb25FdmVudBIXCg9jb2xsZWN0aW9uX25hbWUYASABKAkSFAoMc291cmNlX2luZGV4GAIgASgJEjYKBnJlcG9ydBgDIAEoCzImLnR1cmJvLnNlYXJjaC52MS5Wb2NhYnVsYXJ5RHJpZnRSZXBvcnQqWgoRVmVjdG9yU3RvcmFnZUtpbmQSIAocVkVDVE9SX1NUT1JBR0VfS0lORF9GTEFUX0YzMhAAEiMKH1ZFQ1RPUl9TVE9SQUdFX0tJTkRfVFVSQk9fUVVBTlQQASpFCg5EaXN0YW5jZU1ldHJpYxIaChZESVNUQU5DRV9NRVRSSUNfQ09TSU5FEAASFwoTRElTVEFOQ0VfTUVUUklDX0RPVBABKk0KEU5vcm1hbGl6YXRpb25Nb2RlEhsKF05PUk1BTElaQVRJT05fTU9ERV9OT05FEAASGwoXTk9STUFMSVpBVElPTl9NT0RFX1VOSVQQASpBCg1UZXJtTWF0Y2hNb2RlEhcKE1RFUk1fTUFUQ0hfTU9ERV9BTlkQABIXChNURVJNX01BVENIX01PREVfQUxMEAEqOwoMSm9pbk9wZXJhdG9yEhQKEEpPSU5fT1BFUkFUT1JfT1IQABIVChFKT0lOX09QRVJBVE9SX0FORBABKlAKDkZ1c2lvblN0cmF0ZWd5EhkKFUZVU0lPTl9TVFJBVEVHWV9TQ09SRRAAEiMKH0ZVU0lPTl9TVFJBVEVHWV9SRUNJUFJPQ0FMX1JBTksQATLYDAoSVHVyYm9TZWFyY2hTZXJ2aWNlEnIKE0xpc3RTZWFyY2hQcm92aWRlcnMSKy50dXJiby5zZWFyY2gudjEuTGlzdFNlYXJjaFByb3ZpZGVyc1JlcXVlc3QaLC50dXJiby5zZWFyY2gudjEuTGlzdFNlYXJjaFByb3ZpZGVyc1Jlc3BvbnNlIgASWgoLTGlzdEluZGV4ZXMSIy50dXJiby5zZWFyY2gudjEuTGlzdEluZGV4ZXNSZXF1ZXN0GiQudHVyYm8uc2VhcmNoLnYxLkxpc3RJbmRleGVzUmVzcG9uc2UiABJaCgtDcmVhdGVJbmRleBIjLnR1cmJvLnNlYXJjaC52MS5DcmVhdGVJbmRleFJlcXVlc3QaJC50dXJiby5zZWFyY2gudjEuQ3JlYXRlSW5kZXhSZXNwb25zZSIAEloKC0RlbGV0ZUluZGV4EiMudHVyYm8uc2VhcmNoLnYxLkRlbGV0ZUluZGV4UmVxdWVzdBokLnR1cmJvLnNlYXJjaC52MS5EZWxldGVJbmRleFJlc3BvbnNlIgASZAoOSW5kZXhEb2N1bWVudHMSJS50dXJiby5zZWFyY2gudjEuSW5kZXhEb2N1bWVudFJlcXVlc3QaJy50dXJiby5zZWFyY2gudjEuSW5kZXhEb2N1bWVudHNSZXNwb25zZSIAKAESSwoGU2VhcmNoEh4udHVyYm8uc2VhcmNoLnYxLlNlYXJjaFJlcXVlc3QaHy50dXJiby5zZWFyY2gudjEuU2VhcmNoUmVzcG9uc2UiABJdCgxQZXJzaXN0SW5kZXgSJC50dXJiby5zZWFyY2gudjEuUGVyc2lzdEluZGV4UmVxdWVzdBolLnR1cmJvLnNlYXJjaC52MS5QZXJzaXN0SW5kZXhSZXNwb25zZSIAElQKCVNlYWxJbmRleBIhLnR1cmJvLnNlYXJjaC52MS5TZWFsSW5kZXhSZXF1ZXN0GiIudHVyYm8uc2VhcmNoLnYxLlNlYWxJbmRleFJlc3BvbnNlIgASTgoHUmVpbmRleBIfLnR1cmJvLnNlYXJjaC52MS5SZWluZGV4UmVxdWVzdBogLnR1cmJvLnNlYXJjaC52MS5SZWluZGV4UmVzcG9uc2UiABJRCghTZXRBbGlhcxIgLnR1cmJvLnNlYXJjaC52MS5TZXRBbGlhc1JlcXVlc3QaIS50dXJiby5zZWFyY2gudjEuU2V0QWxpYXNSZXNwb25zZSIAEloKC0RlbGV0ZUFsaWFzEiMudHVyYm8uc2VhcmNoLnYxLkRlbGV0ZUFsaWFzUmVxdWVzdBokLnR1cmJvLnNlYXJjaC52MS5EZWxldGVBbGlhc1Jlc3BvbnNlIgASWgoLTGlzdEFsaWFzZXMSIy50dXJiby5zZWFyY2gudjEuTGlzdEFsaWFzZXNSZXF1ZXN0GiQudHVyYm8uc2VhcmNoLnYxLkxpc3RBbGlhc2VzUmVzcG9uc2UiABJgCg1TZXRDb2xsZWN0aW9uEiUudHVyYm8uc2VhcmNoLnYxLlNldENvbGxlY3Rpb25SZXF1ZXN0GiYudHVyYm8uc2VhcmNoLnYxLlNldENvbGxlY3Rpb25SZXNwb25zZSIAEmAKDUdldENvbGxlY3Rpb24SJS50dXJiby5zZWFyY2gudjEuR2V0Q29sbGVjdGlvblJlcXVlc3QaJi50dXJiby5zZWFyY2gudjEuR2V0Q29sbGVjdGlvblJlc3BvbnNlIgASZgoPTGlzdENvbGxlY3Rpb25zEicudHVyYm8uc2VhcmNoLnYxLkxpc3RDb2xsZWN0aW9uc1JlcXVlc3QaKC50dXJiby5zZWFyY2gudjEuTGlzdENvbGxlY3Rpb25zUmVzcG9uc2UiABJpChBEZWxldGVDb2xsZWN0aW9uEigudHVyYm8uc2VhcmNoLnYxLkRlbGV0ZUNvbGxlY3Rpb25SZXF1ZXN0GikudHVyYm8uc2VhcmNoLnYxLkRlbGV0ZUNvbGxlY3Rpb25SZXNwb25zZSIAEmAKD1dhdGNoQ29sbGVjdGlvbhInLnR1cmJvLnNlYXJjaC52MS5XYXRjaENvbGxlY3Rpb25SZXF1ZXN0GiAudHVyYm8uc2VhcmNoLnYxLkNvbGxlY3Rpb25FdmVudCIAMAFiBnByb3RvMw");
var IndexDocumentRequestSchema = /* @__PURE__ */ messageDesc(file_turbo_search, 16);
var VectorStorageKind = /* @__PURE__ */ ((VectorStorageKind2) => {
  VectorStorageKind2[VectorStorageKind2["FLAT_F32"] = 0] = "FLAT_F32";
  VectorStorageKind2[VectorStorageKind2["TURBO_QUANT"] = 1] = "TURBO_QUANT";
  return VectorStorageKind2;
})(VectorStorageKind || {});
var DistanceMetric = /* @__PURE__ */ ((DistanceMetric2) => {
  DistanceMetric2[DistanceMetric2["COSINE"] = 0] = "COSINE";
  DistanceMetric2[DistanceMetric2["DOT"] = 1] = "DOT";
  return DistanceMetric2;
})(DistanceMetric || {});
var TurboSearchService = /* @__PURE__ */ serviceDesc(file_turbo_search, 0);

// demo/web/src/gen/turbo_training_pb.ts
var file_turbo_training = /* @__PURE__ */ fileDesc("ChR0dXJib190cmFpbmluZy5wcm90bxIRdHVyYm8udHJhaW5pbmcudjEiLQoMTGljZW5jZVRlcm1zEg8KB2xpY2VuY2UYASABKAkSDAoEdGV4dBgCIAEoCSJGCg5UcmFpbmluZ1NvdXJjZRIMCgRuYW1lGAEgASgJEg8KB2xpY2VuY2UYAiABKAkSFQoNbGljZW5jZV90ZXJtcxgDIAEoCSKcAQoLQ2F0YWxvZ0ZpbGUSDAoEcGF0aBgBIAEoCRILCgN1cmwYAiABKAkSDAoEc2l6ZRgDIAEoBBIOCgZzaGEyNTYYBCABKAkSDgoGc2hhNTEyGAUgASgJEhMKC2NvbXByZXNzaW9uGAYgASgJEhYKDmRvd25sb2FkX2J5dGVzGAcgASgEEhcKD2Rvd25sb2FkX3NoYTI1NhgIIAEoCSK4AwoMQ2F0YWxvZ0VudHJ5EgoKAmlkGAEgASgJEgwKBGtpbmQYAiABKAkSEQoJY29tcG9uZW50GAMgASgJEhAKCGxhbmd1YWdlGAQgASgJEg4KBmZvcm1hdBgFIAEoCRIPCgd2ZXJzaW9uGAYgASgJEg8KB2xpY2VuY2UYByABKAkSFQoNbGljZW5jZV90ZXJtcxgIIAEoCRIVCg1saWNlbmNlX2NsYXNzGAkgASgJEhMKC2F0dHJpYnV0aW9uGAogASgJEjUKCnRyYWluZWRfb24YCyADKAsyIS50dXJiby50cmFpbmluZy52MS5UcmFpbmluZ1NvdXJjZRItCgVmaWxlcxgMIAMoCzIeLnR1cmJvLnRyYWluaW5nLnYxLkNhdGFsb2dGaWxlEhMKC3RvdGFsX2J5dGVzGA0gASgEEi4KBXRlcm1zGA4gAygLMh8udHVyYm8udHJhaW5pbmcudjEuTGljZW5jZVRlcm1zEjYKDW1pc3NpbmdfdGVybXMYDyADKAsyHy50dXJiby50cmFpbmluZy52MS5MaWNlbmNlVGVybXMSEQoJaW5zdGFsbGVkGBAgASgIIkcKF0xpc3RNb2RlbENhdGFsb2dSZXF1ZXN0EgwKBHVzZXIYASABKAkSDAoEa2luZBgCIAEoCRIQCghsYW5ndWFnZRgDIAEoCSJMChhMaXN0TW9kZWxDYXRhbG9nUmVzcG9uc2USMAoHZW50cmllcxgBIAMoCzIfLnR1cmJvLnRyYWluaW5nLnYxLkNhdGFsb2dFbnRyeSJeCg5JbnN0YWxsZWRNb2RlbBIKCgJpZBgBIAEoCRIPCgd2ZXJzaW9uGAIgASgJEgsKA2RpchgDIAEoCRINCgVmaWxlcxgEIAMoCRITCgt0b3RhbF9ieXRlcxgFIAEoBCIcChpMaXN0SW5zdGFsbGVkTW9kZWxzUmVxdWVzdCJQChtMaXN0SW5zdGFsbGVkTW9kZWxzUmVzcG9uc2USMQoGbW9kZWxzGAEgAygLMiEudHVyYm8udHJhaW5pbmcudjEuSW5zdGFsbGVkTW9kZWwiTgoeUmVjb3JkTGljZW5jZUFjY2VwdGFuY2VSZXF1ZXN0EgwKBHVzZXIYASABKAkSDwoHbGljZW5jZRgCIAEoCRINCgV0ZXJtcxgDIAEoCSJ3Ch9SZWNvcmRMaWNlbmNlQWNjZXB0YW5jZVJlc3BvbnNlEgwKBHVzZXIYASABKAkSMQoIYWNjZXB0ZWQYAiABKAsyHy50dXJiby50cmFpbmluZy52MS5MaWNlbmNlVGVybXMSEwoLYWNjZXB0ZWRfYXQYAyABKAQiOAoTSW5zdGFsbE1vZGVsUmVxdWVzdBIMCgR1c2VyGAEgASgJEhMKC2FydGlmYWN0X2lkGAIgASgJIoYBCg9JbnN0YWxsUHJvZ3Jlc3MSLgoFc3RhZ2UYASABKA4yHy50dXJiby50cmFpbmluZy52MS5JbnN0YWxsU3RhZ2USEgoKZmlsZV9pbmRleBgCIAEoDRISCgpmaWxlX2NvdW50GAMgASgNEgwKBHBhdGgYBCABKAkSDQoFYnl0ZXMYBSABKAQijgEKEkluc3RhbGxNb2RlbFVwZGF0ZRI2Cghwcm9ncmVzcxgBIAEoCzIiLnR1cmJvLnRyYWluaW5nLnYxLkluc3RhbGxQcm9ncmVzc0gAEjYKCWluc3RhbGxlZBgCIAEoCzIhLnR1cmJvLnRyYWluaW5nLnYxLkluc3RhbGxlZE1vZGVsSABCCAoGdXBkYXRlIngKB1RlYWNoZXISDAoEbmFtZRgBIAEoCRIQCghtb2RlbF9pZBgCIAEoCRIQCghyZXZpc2lvbhgDIAEoCRIXCg9tYW5pZmVzdF9zaGEyNTYYBCABKAkSDgoGaGlkZGVuGAUgASgNEhIKCnZvY2FiX3NpemUYBiABKA0iFQoTTGlzdFRlYWNoZXJzUmVxdWVzdCJEChRMaXN0VGVhY2hlcnNSZXNwb25zZRIsCgh0ZWFjaGVycxgBIAMoCzIaLnR1cmJvLnRyYWluaW5nLnYxLlRlYWNoZXIi+QEKF1RyYWluU3RhdGljTW9kZWxSZXF1ZXN0EgwKBG5hbWUYASABKAkSDwoHdGVhY2hlchgCIAEoCRIQCghwY2FfZGltcxgDIAEoDRIXCg9zaWZfY29lZmZpY2llbnQYBCABKAESLQoFZHR5cGUYBSABKA4yHi50dXJiby50cmFpbmluZy52MS5TdGF0aWNEdHlwZRISCgptYXhfbGVuZ3RoGAYgASgNEh4KFnZvY2FidWxhcnlfYXJ0aWZhY3RfaWQYByABKAkSDQoFdGVybXMYCCADKAkSEAoIcmV2aXNpb24YCSABKAkSEAoIc2Vzc2lvbnMYCiABKA0i9gEKGFRyYWluU3RhdGljTW9kZWxQcm9ncmVzcxIsCgVzdGFnZRgBIAEoDjIdLnR1cmJvLnRyYWluaW5nLnYxLlRyYWluU3RhZ2USFwoPY29tcGxldGVkX2l0ZW1zGAIgASgEEhMKC3RvdGFsX2l0ZW1zGAMgASgEEhMKC2FkZGVkX3Rlcm1zGAQgASgEEhUKDXJlZnVzZWRfdGVybXMYBSABKAQSGgoSZXhwbGFpbmVkX3ZhcmlhbmNlGAYgASgBEjYKB3F1YWxpdHkYByABKAsyJS50dXJiby50cmFpbmluZy52MS5TdGF0aWNNb2RlbFF1YWxpdHkiUAoRVGVhY2hlclByb3ZlbmFuY2USEAoIbW9kZWxfaWQYASABKAkSEAoIcmV2aXNpb24YAiABKAkSFwoPbWFuaWZlc3Rfc2hhMjU2GAMgASgJIlkKElN0YXRpY01vZGVsUXVhbGl0eRIRCgliYXNlX3RvcDEYASABKAESEwoLc3RhdGljX3RvcDEYAiABKAESGwoTc2ltaWxhcml0eV9zcGVhcm1hbhgDIAEoASI6CgxBcnRpZmFjdEZpbGUSDAoEcGF0aBgBIAEoCRIMCgRzaXplGAIgASgEEg4KBnNoYTI1NhgDIAEoCSL8AgoVU3RhdGljTW9kZWxEZXNjcmlwdG9yEgwKBG5hbWUYASABKAkSEAoIbW9kZWxfaWQYAiABKAkSEAoIcmV2aXNpb24YAyABKAkSFwoPbWFuaWZlc3Rfc2hhMjU2GAQgASgJEgwKBGRpbXMYBSABKA0SEgoKdm9jYWJfc2l6ZRgGIAEoDRIVCg1sZWFybmVkX3Rlcm1zGAcgASgNEg0KBWR0eXBlGAggASgJEjUKB3RlYWNoZXIYCSABKAsyJC50dXJiby50cmFpbmluZy52MS5UZWFjaGVyUHJvdmVuYW5jZRI2CgdxdWFsaXR5GAogASgLMiUudHVyYm8udHJhaW5pbmcudjEuU3RhdGljTW9kZWxRdWFsaXR5Ei4KBWZpbGVzGAsgAygLMh8udHVyYm8udHJhaW5pbmcudjEuQXJ0aWZhY3RGaWxlEhAKCHZlcmlmaWVkGAwgASgIEg8KB3Byb2JsZW0YDSABKAkSDgoGc2VydmVkGA4gASgIIisKC1JlZnVzZWRUZXJtEgwKBHRlcm0YASABKAkSDgoGcmVhc29uGAIgASgJItQBChJUcmFpbmVkU3RhdGljTW9kZWwSNwoFbW9kZWwYASABKAsyKC50dXJiby50cmFpbmluZy52MS5TdGF0aWNNb2RlbERlc2NyaXB0b3ISEwoLYWRkZWRfdGVybXMYAiADKAkSNQoNcmVmdXNlZF90ZXJtcxgDIAMoCzIeLnR1cmJvLnRyYWluaW5nLnYxLlJlZnVzZWRUZXJtEh4KFnZvY2FidWxhcnlfYXJ0aWZhY3RfaWQYBCABKAkSGQoRdm9jYWJ1bGFyeV9zaGEyNTYYBSABKAkinQEKFlRyYWluU3RhdGljTW9kZWxVcGRhdGUSPwoIcHJvZ3Jlc3MYASABKAsyKy50dXJiby50cmFpbmluZy52MS5UcmFpblN0YXRpY01vZGVsUHJvZ3Jlc3NIABI4Cgd0cmFpbmVkGAIgASgLMiUudHVyYm8udHJhaW5pbmcudjEuVHJhaW5lZFN0YXRpY01vZGVsSABCCAoGdXBkYXRlIhkKF0xpc3RTdGF0aWNNb2RlbHNSZXF1ZXN0IlQKGExpc3RTdGF0aWNNb2RlbHNSZXNwb25zZRI4CgZtb2RlbHMYASADKAsyKC50dXJiby50cmFpbmluZy52MS5TdGF0aWNNb2RlbERlc2NyaXB0b3IiKAoYRGVsZXRlU3RhdGljTW9kZWxSZXF1ZXN0EgwKBG5hbWUYASABKAkiLQoZRGVsZXRlU3RhdGljTW9kZWxSZXNwb25zZRIQCgh1bmxvYWRlZBgBIAEoCCqiAQoMSW5zdGFsbFN0YWdlEh0KGUlOU1RBTExfU1RBR0VfVU5TUEVDSUZJRUQQABIaChZJTlNUQUxMX1NUQUdFX0FETUlUVEVEEAESHQoZSU5TVEFMTF9TVEFHRV9ET1dOTE9BRElORxACEhwKGElOU1RBTExfU1RBR0VfRE9XTkxPQURFRBADEhoKFklOU1RBTExfU1RBR0VfVkVSSUZJRUQQBCpXCgtTdGF0aWNEdHlwZRIcChhTVEFUSUNfRFRZUEVfVU5TUEVDSUZJRUQQABIUChBTVEFUSUNfRFRZUEVfRjMyEAESFAoQU1RBVElDX0RUWVBFX0YxNhACKtcBCgpUcmFpblN0YWdlEhsKF1RSQUlOX1NUQUdFX1VOU1BFQ0lGSUVEEAASGAoUVFJBSU5fU1RBR0VfQURNSVRURUQQARIUChBUUkFJTl9TVEFHRV9LRVBUEAISGQoVVFJBSU5fU1RBR0VfRU1CRURESU5HEAMSFwoTVFJBSU5fU1RBR0VfUkVEVUNFRBAEEhgKFFRSQUlOX1NUQUdFX01FQVNVUkVEEAUSFgoSVFJBSU5fU1RBR0VfU0VBTEVEEAYSFgoSVFJBSU5fU1RBR0VfTE9BREVEEAcy3AMKE1R1cmJvQ2F0YWxvZ1NlcnZpY2USawoQTGlzdE1vZGVsQ2F0YWxvZxIqLnR1cmJvLnRyYWluaW5nLnYxLkxpc3RNb2RlbENhdGFsb2dSZXF1ZXN0GisudHVyYm8udHJhaW5pbmcudjEuTGlzdE1vZGVsQ2F0YWxvZ1Jlc3BvbnNlEnQKE0xpc3RJbnN0YWxsZWRNb2RlbHMSLS50dXJiby50cmFpbmluZy52MS5MaXN0SW5zdGFsbGVkTW9kZWxzUmVxdWVzdBouLnR1cmJvLnRyYWluaW5nLnYxLkxpc3RJbnN0YWxsZWRNb2RlbHNSZXNwb25zZRKAAQoXUmVjb3JkTGljZW5jZUFjY2VwdGFuY2USMS50dXJiby50cmFpbmluZy52MS5SZWNvcmRMaWNlbmNlQWNjZXB0YW5jZVJlcXVlc3QaMi50dXJiby50cmFpbmluZy52MS5SZWNvcmRMaWNlbmNlQWNjZXB0YW5jZVJlc3BvbnNlEl8KDEluc3RhbGxNb2RlbBImLnR1cmJvLnRyYWluaW5nLnYxLkluc3RhbGxNb2RlbFJlcXVlc3QaJS50dXJiby50cmFpbmluZy52MS5JbnN0YWxsTW9kZWxVcGRhdGUwATLBAwoUVHVyYm9UcmFpbmluZ1NlcnZpY2USXwoMTGlzdFRlYWNoZXJzEiYudHVyYm8udHJhaW5pbmcudjEuTGlzdFRlYWNoZXJzUmVxdWVzdBonLnR1cmJvLnRyYWluaW5nLnYxLkxpc3RUZWFjaGVyc1Jlc3BvbnNlEmsKEFRyYWluU3RhdGljTW9kZWwSKi50dXJiby50cmFpbmluZy52MS5UcmFpblN0YXRpY01vZGVsUmVxdWVzdBopLnR1cmJvLnRyYWluaW5nLnYxLlRyYWluU3RhdGljTW9kZWxVcGRhdGUwARJrChBMaXN0U3RhdGljTW9kZWxzEioudHVyYm8udHJhaW5pbmcudjEuTGlzdFN0YXRpY01vZGVsc1JlcXVlc3QaKy50dXJiby50cmFpbmluZy52MS5MaXN0U3RhdGljTW9kZWxzUmVzcG9uc2USbgoRRGVsZXRlU3RhdGljTW9kZWwSKy50dXJiby50cmFpbmluZy52MS5EZWxldGVTdGF0aWNNb2RlbFJlcXVlc3QaLC50dXJiby50cmFpbmluZy52MS5EZWxldGVTdGF0aWNNb2RlbFJlc3BvbnNlYgZwcm90bzM");
var TurboCatalogService = /* @__PURE__ */ serviceDesc(file_turbo_training, 0);
var TurboTrainingService = /* @__PURE__ */ serviceDesc(file_turbo_training, 1);

// demo/web/src/gen/turbo_vocabulary_pb.ts
var file_turbo_vocabulary = /* @__PURE__ */ fileDesc("ChZ0dXJib192b2NhYnVsYXJ5LnByb3RvEhN0dXJiby52b2NhYnVsYXJ5LnYxIh4KHExpc3REaWN0aW9uYXJ5Rm9ybWF0c1JlcXVlc3QiZgoQRGljdGlvbmFyeUZvcm1hdBIRCglmb3JtYXRfaWQYASABKAkSEwoLZGVzY3JpcHRpb24YAiABKAkSEgoKaW1wb3J0YWJsZRgDIAEoCBIWCg5uZWVkc19sYW5ndWFnZRgEIAEoCCKfAQoLU3RvcmVMaW1pdHMSDQoFc3RvcmUYASABKAgSDQoFYWRtaW4YAiABKAgSIwobbWF4X2RpY3Rpb25hcnlfdXBsb2FkX2J5dGVzGAMgASgEEhcKD21heF9sZWFybl9ieXRlcxgEIAEoBBIWCg5tYXhfbmdyYW1fc2l6ZRgFIAEoDRIcChRkb3dubG9hZF9jaHVua19ieXRlcxgGIAEoDSKJAQodTGlzdERpY3Rpb25hcnlGb3JtYXRzUmVzcG9uc2USNgoHZm9ybWF0cxgBIAMoCzIlLnR1cmJvLnZvY2FidWxhcnkudjEuRGljdGlvbmFyeUZvcm1hdBIwCgZsaW1pdHMYAiABKAsyIC50dXJiby52b2NhYnVsYXJ5LnYxLlN0b3JlTGltaXRzIp4BChxEaWN0aW9uYXJ5QXJ0aWZhY3REZXNjcmlwdG9yEhMKC2FydGlmYWN0X2lkGAEgASgJEhEKCWZvcm1hdF9pZBgCIAEoCRIQCghsYW5ndWFnZRgDIAEoCRIMCgRuYW1lGAQgASgJEhMKC2VudHJ5X2NvdW50GAUgASgEEhEKCWJ5dGVfc2l6ZRgGIAEoBBIOCgZzaGEyNTYYByABKAki2QEKHFZvY2FidWxhcnlBcnRpZmFjdERlc2NyaXB0b3ISEwoLYXJ0aWZhY3RfaWQYASABKAkSEAoIbGFuZ3VhZ2UYAiABKAkSEQoJZG9jdW1lbnRzGAMgASgEEg4KBnRva2VucxgEIAEoBBISCgp0ZXJtX2NvdW50GAUgASgEEhMKC25ncmFtX2NvdW50GAYgASgEEhEKCWJ5dGVfc2l6ZRgHIAEoBBIOCgZzaGEyNTYYCCABKAkSIwobc2VlZF9kaWN0aW9uYXJ5X2FydGlmYWN0X2lkGAkgASgJIhkKF0xpc3REaWN0aW9uYXJpZXNSZXF1ZXN0ImMKGExpc3REaWN0aW9uYXJpZXNSZXNwb25zZRJHCgxkaWN0aW9uYXJpZXMYASADKAsyMS50dXJiby52b2NhYnVsYXJ5LnYxLkRpY3Rpb25hcnlBcnRpZmFjdERlc2NyaXB0b3IiGQoXTGlzdFZvY2FidWxhcmllc1JlcXVlc3QiYwoYTGlzdFZvY2FidWxhcmllc1Jlc3BvbnNlEkcKDHZvY2FidWxhcmllcxgBIAMoCzIxLnR1cmJvLnZvY2FidWxhcnkudjEuVm9jYWJ1bGFyeUFydGlmYWN0RGVzY3JpcHRvciI2Ch9EZWxldGVWb2NhYnVsYXJ5QXJ0aWZhY3RSZXF1ZXN0EhMKC2FydGlmYWN0X2lkGAEgASgJIiIKIERlbGV0ZVZvY2FidWxhcnlBcnRpZmFjdFJlc3BvbnNlIoYBChZJbXBvcnREaWN0aW9uYXJ5SGVhZGVyEhEKCWZvcm1hdF9pZBgBIAEoCRIQCghsYW5ndWFnZRgCIAEoCRIMCgRuYW1lGAMgASgJEhEKCWJ5dGVfc2l6ZRgEIAEoBBIOCgZzaGEyNTYYBSABKAkSFgoOaGFzX2hlYWRlcl9yb3cYBiABKAgicgoXSW1wb3J0RGljdGlvbmFyeVJlcXVlc3QSPQoGaGVhZGVyGAEgASgLMisudHVyYm8udm9jYWJ1bGFyeS52MS5JbXBvcnREaWN0aW9uYXJ5SGVhZGVySAASDwoFY2h1bmsYAiABKAxIAEIHCgVmcmFtZSL+AQoVTGVhcm5Wb2NhYnVsYXJ5SGVhZGVyEhoKEm1pbl90ZXJtX2ZyZXF1ZW5jeRgBIAEoDRIRCgltYXhfdGVybXMYAiABKA0SEQoJbG93ZXJjYXNlGAMgASgIEhgKEGZpbHRlcl9zdG9wd29yZHMYBCABKAgSFgoOaW5jbHVkZV9uZ3JhbXMYBSABKAgSFgoObWluX25ncmFtX3NpemUYBiABKA0SFgoObWF4X25ncmFtX3NpemUYByABKA0SDwoHbWluX3BtaRgIIAEoARIeChZkaWN0aW9uYXJ5X2FydGlmYWN0X2lkGAkgASgJEhAKCGxhbmd1YWdlGAogASgJIicKF0xlYXJuVm9jYWJ1bGFyeURvY3VtZW50EgwKBHRleHQYASABKAkioQEKFkxlYXJuVm9jYWJ1bGFyeVJlcXVlc3QSPAoGaGVhZGVyGAEgASgLMioudHVyYm8udm9jYWJ1bGFyeS52MS5MZWFyblZvY2FidWxhcnlIZWFkZXJIABJACghkb2N1bWVudBgCIAEoCzIsLnR1cmJvLnZvY2FidWxhcnkudjEuTGVhcm5Wb2NhYnVsYXJ5RG9jdW1lbnRIAEIHCgVmcmFtZSIwChlEb3dubG9hZFZvY2FidWxhcnlSZXF1ZXN0EhMKC2FydGlmYWN0X2lkGAEgASgJIsQBChdWb2NhYnVsYXJ5QXJ0aWZhY3RDaHVuaxJHCgpkaWN0aW9uYXJ5GAEgASgLMjEudHVyYm8udm9jYWJ1bGFyeS52MS5EaWN0aW9uYXJ5QXJ0aWZhY3REZXNjcmlwdG9ySAASRwoKdm9jYWJ1bGFyeRgCIAEoCzIxLnR1cmJvLnZvY2FidWxhcnkudjEuVm9jYWJ1bGFyeUFydGlmYWN0RGVzY3JpcHRvckgAEg4KBGRhdGEYAyABKAxIAEIHCgVmcmFtZTL1BgoWVHVyYm9Wb2NhYnVsYXJ5U2VydmljZRKAAQoVTGlzdERpY3Rpb25hcnlGb3JtYXRzEjEudHVyYm8udm9jYWJ1bGFyeS52MS5MaXN0RGljdGlvbmFyeUZvcm1hdHNSZXF1ZXN0GjIudHVyYm8udm9jYWJ1bGFyeS52MS5MaXN0RGljdGlvbmFyeUZvcm1hdHNSZXNwb25zZSIAEnEKEExpc3REaWN0aW9uYXJpZXMSLC50dXJiby52b2NhYnVsYXJ5LnYxLkxpc3REaWN0aW9uYXJpZXNSZXF1ZXN0Gi0udHVyYm8udm9jYWJ1bGFyeS52MS5MaXN0RGljdGlvbmFyaWVzUmVzcG9uc2UiABJxChBMaXN0Vm9jYWJ1bGFyaWVzEiwudHVyYm8udm9jYWJ1bGFyeS52MS5MaXN0Vm9jYWJ1bGFyaWVzUmVxdWVzdBotLnR1cmJvLnZvY2FidWxhcnkudjEuTGlzdFZvY2FidWxhcmllc1Jlc3BvbnNlIgASiQEKGERlbGV0ZVZvY2FidWxhcnlBcnRpZmFjdBI0LnR1cmJvLnZvY2FidWxhcnkudjEuRGVsZXRlVm9jYWJ1bGFyeUFydGlmYWN0UmVxdWVzdBo1LnR1cmJvLnZvY2FidWxhcnkudjEuRGVsZXRlVm9jYWJ1bGFyeUFydGlmYWN0UmVzcG9uc2UiABJ3ChBJbXBvcnREaWN0aW9uYXJ5EiwudHVyYm8udm9jYWJ1bGFyeS52MS5JbXBvcnREaWN0aW9uYXJ5UmVxdWVzdBoxLnR1cmJvLnZvY2FidWxhcnkudjEuRGljdGlvbmFyeUFydGlmYWN0RGVzY3JpcHRvciIAKAESdQoPTGVhcm5Wb2NhYnVsYXJ5EisudHVyYm8udm9jYWJ1bGFyeS52MS5MZWFyblZvY2FidWxhcnlSZXF1ZXN0GjEudHVyYm8udm9jYWJ1bGFyeS52MS5Wb2NhYnVsYXJ5QXJ0aWZhY3REZXNjcmlwdG9yIgAoARJ2ChJEb3dubG9hZFZvY2FidWxhcnkSLi50dXJiby52b2NhYnVsYXJ5LnYxLkRvd25sb2FkVm9jYWJ1bGFyeVJlcXVlc3QaLC50dXJiby52b2NhYnVsYXJ5LnYxLlZvY2FidWxhcnlBcnRpZmFjdENodW5rIgAwAWIGcHJvdG8z");
var ImportDictionaryHeaderSchema = /* @__PURE__ */ messageDesc(file_turbo_vocabulary, 12);
var LearnVocabularyHeaderSchema = /* @__PURE__ */ messageDesc(file_turbo_vocabulary, 14);
var LearnVocabularyDocumentSchema = /* @__PURE__ */ messageDesc(file_turbo_vocabulary, 15);
var TurboVocabularyService = /* @__PURE__ */ serviceDesc(file_turbo_vocabulary, 0);

// demo/web/src/rpc.ts
var TOKEN_KEY = "turbo-demo-token";
function token() {
  return sessionStorage.getItem(TOKEN_KEY) ?? "";
}
function setToken(t) {
  if (t) sessionStorage.setItem(TOKEN_KEY, t);
  else sessionStorage.removeItem(TOKEN_KEY);
}
var bearer = (next) => async (req) => {
  const t = token();
  if (t) req.header.set("authorization", `Bearer ${t}`);
  return next(req);
};
function base() {
  return location.origin;
}
var transport = createConnectTransport({
  baseUrl: base(),
  useBinaryFormat: true,
  useHttpGet: false,
  interceptors: [bearer]
});
function client(service) {
  return createClient(service, transport);
}
var inference = client(GRPCInferenceService);
var analysis = client(AnalysisService);
var search = client(TurboSearchService);
var vocabulary = client(TurboVocabularyService);
var catalog = client(TurboCatalogService);
var training = client(TurboTrainingService);
function codeName(code) {
  const camel = Code[code] ?? String(code);
  let out = "";
  for (const ch of camel) {
    const lower = ch.toLowerCase();
    if (ch !== lower && out) out += "_";
    out += lower;
  }
  return out;
}
function describe(e) {
  if (e instanceof Error && e.name === "AbortError") return "cancelled";
  const c = ConnectError.from(e);
  if (c.code === Code.Canceled) return "cancelled";
  const parts = [codeName(c.code)];
  const tc = c.metadata.get("turbo-code");
  const tf = c.metadata.get("turbo-field");
  if (tc) parts.push(`turbo-code ${tc}`);
  if (tf && tf !== "0") parts.push(`field ${tf}`);
  const why = refusedWhy(c.code);
  return `${parts.join(", ")}: ${c.rawMessage}${why ? ` (${why})` : ""}`;
}
function refusedWhy(code) {
  if (code === Code.PermissionDenied) return "not allowed: start the demo with --allow for this method";
  if (code === Code.Unauthenticated) return "the demo wants its token: set it in the header";
  if (code === Code.Unimplemented) return "not served";
  return "";
}

// demo/web/src/state.ts
var current = null;
var listeners = [];
function capabilities() {
  if (!current) {
    current = analysis.capabilities({});
    current.then((c) => listeners.forEach((l) => l(c))).catch(() => void 0);
  }
  return current;
}
function refresh() {
  current = null;
  return capabilities();
}
function onCapabilities(l) {
  listeners.push(l);
}
function bundles(c, task) {
  return c.bundles.filter((b) => b.task === task).map((b) => b.name);
}

// demo/web/src/tabs/analyze.ts
var running = null;
var graph = null;
var page = 0;
var filter = /* @__PURE__ */ new Set();
var select = (id) => $(id);
function statusName(s) {
  return LayerWireStatus[s] ?? String(s);
}
function layerRows(layers) {
  return table(
    ["layer", "instance", "status", "items", "annotator"],
    [...layers.values()].map((l) => [
      l.layer,
      l.instance,
      el("span", { class: l.status === 0 /* OK */ ? "" : "error", text: statusName(l.status), title: l.statusDetail }),
      String(l.items),
      `${l.annotator} ${l.version}`.trim()
    ])
  );
}
function stop() {
  running?.abort();
  running = null;
}
async function run() {
  stop();
  const ac = new AbortController();
  running = ac;
  const note = $("analyze-note");
  const steps = $("analyze-steps");
  const layersBox = $("analyze-layers");
  steps.replaceChildren();
  layersBox.replaceChildren();
  $("analyze-window").replaceChildren();
  $("analyze-export-out").replaceChildren();
  graph = null;
  const layers = /* @__PURE__ */ new Map();
  const t0 = performance.now();
  say(note, "Analyzing\u2026", "busy");
  try {
    const req = {
      docId: "page",
      text: document.getElementById("analyze-text").value,
      profile: select("analyze-profile").value,
      model: select("analyze-model").value,
      offsets: 1 /* UTF16 */
    };
    for await (const ev of analysis.analyzeDocumentProgressive(req, { signal: ac.signal })) {
      const ms = (performance.now() - t0).toFixed(1);
      const e = ev.event;
      if (e.case === "started") {
        steps.append(el("li", { class: "muted", text: `started: ${e.value.stepIds.join(" \u2192 ")}` }));
      } else if (e.case === "layersReady") {
        const b = e.value;
        for (const l of b.layers) layers.set(`${l.layer}#${l.instance}`, l);
        steps.append(
          el(
            "li",
            {},
            el("span", { class: "badge", text: `${(Number(b.durationUs) / 1e3).toFixed(2)} ms` }),
            ` ${b.completedStep} (stage ${b.stage}, ${b.layers.length} layers, at ${ms} ms)`
          )
        );
        if (b.graph.length) graph = b.graph;
        layersBox.replaceChildren(layerRows(layers));
      } else if (e.case === "stepFailed") {
        const f = e.value;
        steps.append(
          el("li", { class: "error", text: `${f.stepId} failed: ${f.failure?.message ?? ""}; skipped ${f.skippedDependents.join(", ") || "nothing"}` })
        );
      } else if (e.case === "complete") {
        graph = e.value.graph;
        for (const l of e.value.layers) layers.set(`${l.layer}#${l.instance}`, l);
        layersBox.replaceChildren(layerRows(layers));
        steps.append(el("li", { class: "muted", text: `complete at ${ms} ms, ${graph.length} graph bytes` }));
      }
    }
    say(note, "");
    page = 0;
    filter = /* @__PURE__ */ new Set();
    if (graph) await showWindow();
  } catch (e) {
    say(note, describe(e), "error");
  } finally {
    if (running === ac) running = null;
  }
}
function segmentsView(w) {
  const text = el("div", { class: "doc" });
  for (const s of w.segments) {
    if (!s.annotations.length) {
      text.append(s.text);
      continue;
    }
    const tip = s.annotations.map((a) => `${a.layer}${a.label ? `: ${a.label}` : ""}${a.score ? ` ${a.score.toFixed(3)}` : ""}`).join("\n");
    const level = Math.min(4, s.annotations.length);
    text.append(el("mark", { class: `l${level}`, title: tip, text: s.text }));
  }
  return text;
}
function lanesView(w) {
  const box = el("div", { class: "lanes" });
  for (const lane of w.heatmapLanes) {
    const top = Math.max(1, ...lane.buckets.map((b) => Math.abs(b.score) || b.count));
    box.append(
      el(
        "div",
        { class: "lane" },
        el("span", { class: "lane-name", text: `${lane.id} (${lane.kind})` }),
        el(
          "span",
          { class: "cells" },
          lane.buckets.map((b) => {
            const v = (Math.abs(b.score) || b.count) / top;
            return el("span", { class: "hcell", title: `${b.start}..${b.end}: ${b.count} items, score ${b.score.toFixed(3)}`, style: `opacity: ${(0.1 + 0.9 * v).toFixed(2)}` });
          })
        )
      )
    );
  }
  return box;
}
async function showWindow() {
  if (!graph) return;
  const box = $("analyze-window");
  try {
    const base2 = { graph, view: 0, offsets: 1 /* UTF16 */, pageSize: 16e3, filterLayers: [...filter] };
    const [w, heat2] = await Promise.all([
      analysis.inspectDocumentWindow({ ...base2, page }),
      analysis.inspectDocumentWindow({ ...base2, page: 0, heatmapBuckets: 96 })
    ]);
    const win = w.window;
    const chips = (w.summary?.layers ?? []).map((l) => {
      const on = filter.has(l.layer);
      const b = el("button", { type: "button", class: on ? "chip on" : "chip", text: `${l.layer} ${l.items}` });
      b.addEventListener("click", () => {
        if (on) filter.delete(l.layer);
        else filter.add(l.layer);
        void showWindow();
      });
      return b;
    });
    const nav = el("div", { class: "nav" });
    const prev = el("button", { type: "button", text: "\u2190 page" });
    const next = el("button", { type: "button", text: "page \u2192" });
    prev.toggleAttribute("disabled", !win || win.page === 0);
    next.toggleAttribute("disabled", !win || win.page + 1 >= win.pageCount);
    prev.addEventListener("click", () => (page = Math.max(0, page - 1), void showWindow()));
    next.addEventListener("click", () => (page += 1, void showWindow()));
    nav.append(prev, el("span", { class: "muted", text: win ? ` page ${win.page + 1} of ${win.pageCount}, units ${win.start}..${win.end} ` : "" }), next);
    const s = w.summary;
    box.replaceChildren(
      el("h3", { text: "Document" }),
      s ? el("p", { class: "muted", text: `${s.totalBytes} bytes, ${s.totalUnits} UTF-16 units, ${s.viewCount} views, ${s.layerCount} layers. Pick layers to show only them.` }) : "",
      el("div", { class: "chips" }, chips),
      nav,
      segmentsView(w),
      el("h3", { text: "Heatmap" }),
      lanesView(heat2)
    );
  } catch (e) {
    box.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}
async function exportDoc() {
  const out = $("analyze-export-out");
  if (!graph) {
    out.replaceChildren(el("p", { class: "error", text: "Analyze a document first." }));
    return;
  }
  try {
    const r = await analysis.formatDocument({ docId: "page", graph, formatId: select("analyze-format").value, offsets: 1 /* UTF16 */ });
    const name = `document${r.fileExtension.startsWith(".") ? "" : "."}${r.fileExtension}`;
    const save = el("button", { type: "button", text: `Download ${name}` });
    save.addEventListener("click", () => download(name, [r.content], r.mediaType));
    const text = r.mediaType.startsWith("text/") || r.mediaType.includes("csv") || r.mediaType.includes("ld+");
    out.replaceChildren(save, text ? el("pre", { class: "export", text: new TextDecoder().decode(r.content) }) : el("p", { class: "muted", text: `${r.content.length} bytes of ${r.mediaType}` }));
  } catch (e) {
    out.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}
async function loadFormats() {
  try {
    const f = await analysis.listOutputFormats({});
    options(select("analyze-format"), f.formats.map((x) => x.formatId), f.formats.map((x) => `${x.displayName} (${x.mediaType})`));
  } catch (e) {
    say($("analyze-note"), describe(e), "error");
  }
}
function init() {
  $("analyze-run").addEventListener("click", run);
  $("analyze-stop").addEventListener("click", stop);
  $("analyze-export").addEventListener("click", exportDoc);
  onCapabilities((c) => {
    options(select("analyze-profile"), ["", ...c.profiles.map((p) => p.name)], ["default", ...c.profiles.map((p) => `${p.name} v${p.version}`)]);
    options(select("analyze-model"), ["", ...bundles(c, "TASK_EMBED")], ["none", ...bundles(c, "TASK_EMBED")]);
  });
  capabilities().catch(() => void 0);
  void loadFormats();
}

// demo/web/src/tabs/embed.ts
var models = /* @__PURE__ */ new Map();
function tier(meta) {
  const p = meta?.properties["session_info.precision"];
  return p ? after(p, "PRECISION_") : "?";
}
function facts(meta) {
  const p = meta.properties;
  const out = [];
  if (p["model_info.model_id"]) out.push(p["model_info.model_id"]);
  if (p["model_info.dim"]) out.push(`dim ${p["model_info.dim"]}`);
  if (p["session_info.compute_dtype"]) out.push(after(p["session_info.compute_dtype"], "DTYPE_"));
  const dev = [p["device_info.backend"], p["device_info.name"]].filter(Boolean).join(" ");
  if (dev) out.push(dev);
  return out.join(" \xB7 ");
}
async function loadMeta(name) {
  const m = models.get(name);
  try {
    m.meta = await inference.modelMetadata({ name });
    m.error = "";
  } catch (e) {
    m.meta = null;
    m.error = describe(e);
    m.checked = false;
  }
}
function renderModels() {
  const box = $("embed-models");
  box.replaceChildren();
  if (models.size === 0) box.append(el("p", { class: "muted", text: "No embedding model is served: add a name below." }));
  for (const [name, m] of models) {
    const cb = el("input", { type: "checkbox" });
    cb.checked = m.checked;
    cb.disabled = !m.meta;
    cb.addEventListener("change", () => m.checked = cb.checked);
    box.append(
      el(
        "label",
        { class: "model" },
        cb,
        el("span", {}, el("span", { class: "name", text: name }), m.meta ? el("span", { class: "tag", text: tier(m.meta) }) : null),
        el("span", { class: m.error ? "facts error" : "facts", text: m.meta ? facts(m.meta) : m.error })
      )
    );
  }
}
async function addModels(names) {
  const fresh = names.filter((n) => !models.has(n));
  for (const n of fresh) models.set(n, { meta: null, error: "", checked: true });
  await Promise.all(fresh.map(loadMeta));
  renderModels();
}
function cosine(a, b) {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? d / Math.sqrt(na * nb) : 0;
}
function norm(a) {
  return Math.sqrt(a.reduce((s, x) => s + x * x, 0));
}
function matrix(vectors) {
  return vectors.map((a) => vectors.map((b) => cosine(a, b)));
}
function half(bits) {
  const s = bits & 32768 ? -1 : 1;
  const e = bits >> 10 & 31;
  const f = bits & 1023;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}
function vectorsOf(reply) {
  const out = reply.outputs[0];
  const raw = reply.rawOutputContents[0];
  if (!out || !raw) throw new Error("the reply has no vectors");
  const [rows, dim] = out.shape.map(Number);
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const width = out.datatype === "FP32" ? 4 : out.datatype === "FP16" || out.datatype === "BF16" ? 2 : 0;
  if (!width) throw new Error(`vectors in ${out.datatype} are not shown`);
  if (raw.byteLength !== rows * dim * width) throw new Error(`${raw.byteLength} bytes for a ${rows}x${dim} ${out.datatype} tensor`);
  const v = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < dim; c++) {
      const at = (r * dim + c) * width;
      if (out.datatype === "FP32") row.push(dv.getFloat32(at, true));
      else if (out.datatype === "FP16") row.push(half(dv.getUint16(at, true)));
      else {
        const f = new DataView(new ArrayBuffer(4));
        f.setUint16(0, dv.getUint16(at, true), false);
        row.push(f.getFloat32(0, false));
      }
    }
    v.push(row);
  }
  return v;
}
function param(p) {
  return create(InferParameterSchema, { parameterChoice: p });
}
function parameters() {
  const p = {};
  const role = document.getElementById("embed-role").value;
  if (role) p.prompt_role = param({ case: "stringParam", value: role });
  const max = parseInt(input("embed-max").value, 10);
  if (max > 0) {
    p.max_tokens = param({ case: "int64Param", value: BigInt(max) });
    p.truncate = param({ case: "stringParam", value: "TRUNCATE_RIGHT" });
  }
  return p;
}
async function run2() {
  const texts = document.getElementById("embed-texts").value.split("\n").map((t) => t.trim()).filter((t) => t.length);
  const chosen = [...models].filter(([, m]) => m.checked && m.meta).map(([n]) => n);
  const err = $("embed-error");
  say(err, "");
  if (!texts.length || !chosen.length) {
    say(err, !texts.length ? "Enter at least one text." : "Pick at least one model.", "error");
    return;
  }
  const btn = $("embed-run");
  btn.disabled = true;
  btn.textContent = "Embedding\u2026";
  const encoder = new TextEncoder();
  const results = [];
  for (const name of chosen) {
    const t0 = performance.now();
    const meta = models.get(name).meta;
    try {
      const reply = await inference.modelInfer({
        modelName: name,
        parameters: parameters(),
        inputs: [{ name: "texts", datatype: "BYTES", shape: [BigInt(texts.length)], contents: { bytesContents: texts.map((t) => encoder.encode(t)) } }]
      });
      results.push({ name, meta, ms: performance.now() - t0, reply, vectors: vectorsOf(reply) });
    } catch (e) {
      results.push({ name, meta, ms: performance.now() - t0, error: describe(e) });
    }
  }
  btn.disabled = false;
  btn.textContent = "Embed";
  render(texts, results);
}
function heat(v) {
  const a = Math.max(0, Math.min(1, v));
  return `background: rgba(var(--heat), ${(a * a * 0.85).toFixed(3)})`;
}
function short(t, n = 28) {
  const chars = typeof Intl !== "undefined" && Intl.Segmenter ? Array.from(new Intl.Segmenter(void 0, { granularity: "grapheme" }).segment(t), (g) => g.segment) : Array.from(t);
  return chars.length > n ? chars.slice(0, n - 1).join("") + "\u2026" : t;
}
function card(texts, r) {
  const head = el("div", { class: "head" }, el("h3", {}, r.name, el("span", { class: "tag", text: tier(r.meta) })));
  if (!r.vectors || !r.reply) {
    return el("div", { class: "card" }, head, el("p", { class: "error", text: r.error ?? "" }), stat(`${r.ms.toFixed(1)} ms`, "browser round trip"));
  }
  const v = r.vectors;
  const p = r.reply.parameters;
  const str = (k) => p[k]?.parameterChoice.case === "stringParam" ? String(p[k].parameterChoice.value) : "";
  const sim = matrix(v);
  const simTable = el(
    "table",
    { class: "sim" },
    el("tr", {}, el("th", {}), texts.map((_, j) => el("th", { text: `#${j + 1}` }))),
    texts.map(
      (t, i) => el(
        "tr",
        {},
        el("td", { class: "text", title: t, text: `#${i + 1} ${short(t, 22)}` }),
        sim[i].map((x) => el("td", { class: "cell", style: heat(x), text: x.toFixed(3) }))
      )
    )
  );
  const previews = el(
    "table",
    {},
    el("tr", {}, el("th", { class: "text", text: "text" }), el("th", { text: "norm" }), el("th", { class: "text", text: "first values" })),
    v.map(
      (row, i) => el(
        "tr",
        {},
        el("td", { class: "text", title: texts[i], text: `#${i + 1} ${short(texts[i], 18)}` }),
        el("td", { text: norm(row).toFixed(4) }),
        el("td", {
          class: "vec",
          text: row.slice(0, 6).map((x) => (x < 0 ? "" : " ") + x.toFixed(4)).join(" ") + (row.length > 6 ? "  \u2026" : "")
        })
      )
    )
  );
  const device = [str("backend"), str("arch")].filter(Boolean).join(" ");
  return el(
    "div",
    { class: "card" },
    head,
    el(
      "div",
      { class: "stats" },
      stat(String(v[0]?.length ?? 0), "dimensions"),
      stat(String(v.length), "texts"),
      stat(`${r.ms.toFixed(1)} ms`, "browser round trip"),
      device ? stat(device, "ran on") : null,
      str("compute_dtype") ? stat(after(str("compute_dtype"), "DTYPE_"), "computed in") : null
    ),
    el("h2", { text: "Cosine similarity" }),
    simTable,
    el("details", {}, el("summary", { text: "Vectors" }), previews)
  );
}
function comparison(texts, ok) {
  const sims = ok.map((r) => matrix(r.vectors));
  const rows = [];
  let widest = 0;
  for (let i = 0; i < texts.length; i++) {
    for (let j = i + 1; j < texts.length; j++) {
      const vals = sims.map((m) => m[i][j]);
      const spread = Math.max(...vals) - Math.min(...vals);
      widest = Math.max(widest, spread);
      rows.push(
        el(
          "tr",
          {},
          el("td", { class: "text", title: `${texts[i]}
${texts[j]}`, text: `#${i + 1} \xB7 #${j + 1}` }),
          vals.map((x) => el("td", { class: "cell", style: heat(x), text: x.toFixed(4) })),
          el("td", { text: spread === 0 ? "0" : spread.toExponential(1) })
        )
      );
    }
  }
  const baseV = ok[0].vectors;
  const agree = ok.map((r) => {
    const v = r.vectors;
    if (v[0].length !== baseV[0].length) return "n/a";
    return Math.min(...v.map((row, i) => cosine(row, baseV[i]))).toFixed(6);
  });
  const tableEl = el(
    "table",
    { class: "sim cmp" },
    el("tr", {}, el("th", { class: "text", text: "pair" }), ok.map((r) => el("th", { text: `${r.name} (${tier(r.meta)})` })), el("th", { text: "spread" })),
    rows,
    el("tr", {}, el("th", { class: "text", text: `lowest cosine to ${ok[0].name}, same text` }), agree.map((a) => el("td", { text: a })), el("td", {}))
  );
  return el(
    "div",
    { class: "card" },
    el("h3", { text: "Side by side" }),
    el("p", {
      class: "muted",
      text: "Each pair of texts, scored by every model picked. The spread is how much the tiers disagree; the last row is how close each model's vector for a text is to the first model's."
    }),
    tableEl,
    widest === 0 ? el("p", { class: "note", text: "Every model gave the same numbers: on this device these tiers compute alike. A device that runs FASTEST in a narrower type shows the difference here." }) : null
  );
}
function render(texts, results) {
  const out = $("embed-results");
  out.replaceChildren();
  const ok = results.filter((r) => r.vectors);
  if (ok.length > 1 && texts.length > 1) out.append(comparison(texts, ok));
  out.append(el("div", { class: "cards" }, results.map((r) => card(texts, r))));
}
function init2() {
  $("embed-add").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const name = input("embed-add-name").value.trim();
    if (!name) return;
    await addModels([name]);
    input("embed-add-name").value = "";
  });
  $("embed-run").addEventListener("click", run2);
  onCapabilities((c) => void addModels(bundles(c, "TASK_EMBED")));
  capabilities().catch((e) => say($("embed-error"), describe(e), "error"));
}

// demo/web/src/frames.ts
var MAX_FRAME_BYTES = (64 << 20) - 16;
var MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024;
function bytes(b) {
  return b;
}
function prefix(n) {
  if (n > MAX_FRAME_BYTES) throw new RangeError(`a frame of ${n} bytes is over the ${MAX_FRAME_BYTES} byte limit`);
  const p = new Uint8Array(5);
  new DataView(p.buffer).setUint32(1, n, false);
  return p;
}
function envelope(message) {
  const out = new Uint8Array(5 + message.length);
  out.set(prefix(message.length));
  out.set(message, 5);
  return out;
}
function varint(n) {
  const out = [];
  let v = n;
  while (v >= 128) {
    out.push(v % 128 | 128);
    v = Math.floor(v / 128);
  }
  out.push(v);
  return Uint8Array.from(out);
}
function fieldHead(field, n) {
  const tag = varint(field * 8 + 2);
  const len = varint(n);
  const out = new Uint8Array(tag.length + len.length);
  out.set(tag);
  out.set(len, tag.length);
  return out;
}
function textFrame(head, field, text) {
  const fh = fieldHead(field, text.size);
  return [prefix(head.length + fh.length + text.size), head, fh, text];
}
function body(parts) {
  const b = new Blob(parts, { type: "application/octet-stream" });
  if (b.size > MAX_UPLOAD_BYTES) throw new RangeError(`the upload is ${b.size} bytes, over the 2 GiB cap`);
  return b;
}
function importBody(header, file) {
  return body([envelope(toBinary(ImportDictionaryHeaderSchema, create(ImportDictionaryHeaderSchema, header))), file]);
}
function learnBody(header, docs) {
  const parts = [envelope(toBinary(LearnVocabularyHeaderSchema, create(LearnVocabularyHeaderSchema, header)))];
  for (const d of docs) {
    if (typeof d === "string") {
      parts.push(envelope(toBinary(LearnVocabularyDocumentSchema, create(LearnVocabularyDocumentSchema, { text: d }))));
    } else {
      parts.push(...textFrame(new Uint8Array(0), 1, d));
    }
  }
  return body(parts);
}
function indexBody(index, docs) {
  const parts = [];
  for (const d of docs) {
    if (typeof d.text === "string") {
      const m = create(IndexDocumentRequestSchema, { indexName: index, docId: d.id, text: d.text });
      parts.push(envelope(toBinary(IndexDocumentRequestSchema, m)));
    } else {
      const head = bytes(toBinary(IndexDocumentRequestSchema, create(IndexDocumentRequestSchema, { indexName: index, docId: d.id })));
      parts.push(...textFrame(head, 3, d.text));
    }
  }
  return body(parts);
}

// demo/web/src/upload.ts
async function put(method, bodyBlob, signal) {
  const headers = new Headers({ "content-type": "application/octet-stream" });
  const t = token();
  if (t) headers.set("authorization", `Bearer ${t}`);
  const url = `${base()}/upload/${method.parent.typeName}/${method.name}`;
  const r = await fetch(url, { method: "PUT", headers, body: bodyBlob, signal });
  if (r.status === 200) return fromBinary(method.output, new Uint8Array(await r.arrayBuffer()));
  throw refusal(r);
}
function refusal(r) {
  const status2 = Number(r.headers.get("grpc-status") ?? "");
  const code = status2 > 0 && status2 <= 16 ? status2 : Code.Unknown;
  let message = r.headers.get("grpc-message") ?? `HTTP ${r.status}`;
  try {
    message = decodeURIComponent(message);
  } catch {
  }
  if (r.status === 415) message = "the demo wants application/octet-stream";
  if (r.status === 413) message = `${message} (HTTP 413: over the 2 GiB cap)`;
  const meta = new Headers();
  for (const k of ["turbo-code", "turbo-field"]) {
    const v = r.headers.get(k);
    if (v !== null) meta.set(k, v);
  }
  return new ConnectError(message, code, meta);
}

// demo/web/src/tabs/search.ts
var caps = null;
var indexing = null;
var select2 = (id) => $(id);
async function list() {
  const box = $("search-indexes");
  try {
    const r = await search.listIndexes({});
    options(select2("search-index"), r.indexes.map((i) => i.indexName));
    box.replaceChildren(
      r.indexes.length ? table(
        ["index", "model", "dim", "metric", "storage", "documents", "chunks", ""],
        r.indexes.map((i) => [
          i.indexName,
          i.space?.model ? `${i.space.model.id}@${i.space.model.revision}` : "",
          String(i.space?.dim ?? ""),
          DistanceMetric[i.metric] ?? "",
          VectorStorageKind[i.storage] ?? "",
          String(i.documentCount),
          String(i.chunkCount),
          deleteButton(i.indexName)
        ])
      ) : el("p", { class: "muted", text: "No index yet: create one from an embedding bundle." })
    );
  } catch (e) {
    box.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}
function deleteButton(name) {
  const b = el("button", { type: "button", text: "Delete" });
  b.addEventListener("click", async () => {
    try {
      await search.deleteIndex({ indexName: name });
      await list();
    } catch (e) {
      say($("search-create-note"), describe(e), "error");
    }
  });
  return b;
}
async function createIndex(ev) {
  ev.preventDefault();
  const note = $("search-create-note");
  const name = input("search-create-name").value.trim();
  const bundle = caps?.bundles.find((b) => b.name === select2("search-create-bundle").value);
  if (!name || !bundle?.model) {
    say(note, "Name the index and pick an embedding bundle.", "error");
    return;
  }
  try {
    await search.createIndex({
      spec: {
        indexName: name,
        space: { model: { id: bundle.model.id, revision: bundle.model.revision, sha256: bundle.model.sha256 }, dim: bundle.widths[0] ?? 0, normalization: 1 /* UNIT */ },
        metric: 0 /* COSINE */,
        storage: 0 /* FLAT_F32 */,
        embedBundle: bundle.name,
        chunkBudgetTokens: 0
      }
    });
    say(note, `Created ${name}.`);
    await list();
    select2("search-index").value = name;
  } catch (e) {
    say(note, describe(e), "error");
  }
}
async function addDocuments() {
  const note = $("search-add-note");
  const index = select2("search-index").value;
  const typed = document.getElementById("search-docs").value.split("\n\n").map((t) => t.trim()).filter((t) => t.length);
  const files = [...input("search-files").files ?? []];
  const docs = [
    ...typed.map((t, i) => ({ id: `typed-${Date.now()}-${i + 1}`, text: t })),
    ...files.map((f) => ({ id: f.name, text: f }))
  ];
  if (!index || !docs.length) {
    say(note, !index ? "Create or pick an index first." : "Type paragraphs (a blank line between) or pick files.", "error");
    return;
  }
  indexing?.abort();
  const ac = new AbortController();
  indexing = ac;
  try {
    const body2 = indexBody(index, docs);
    say(note, `Uploading ${docs.length} documents, ${size(body2.size)}\u2026`, "busy");
    const r = await put(TurboSearchService.method.indexDocuments, body2, ac.signal);
    say(note, `Indexed ${r.indexedDocuments} documents as ${r.indexedChunks} chunks.`);
    await list();
  } catch (e) {
    say(note, describe(e), "error");
  } finally {
    if (indexing === ac) indexing = null;
  }
}
function heatView(r) {
  return el(
    "div",
    {},
    r.documentHeat.map(
      (d) => el(
        "div",
        { class: "lane" },
        el("span", { class: "lane-name", text: `${d.docId} (max ${d.maxScore.toFixed(3)})` }),
        el(
          "span",
          { class: "cells" },
          d.segments.map(
            (s) => el("span", {
              class: "hcell",
              title: `${s.chunkId} ${s.span?.start ?? 0}..${s.span?.end ?? 0}: ${s.score.toFixed(3)}`,
              style: `opacity: ${(0.1 + 0.9 * (d.maxScore ? s.score / d.maxScore : 0)).toFixed(2)}`
            })
          )
        )
      )
    )
  );
}
async function runSearch(ev) {
  ev.preventDefault();
  const out = $("search-results");
  try {
    const r = await search.search({
      indexName: select2("search-index").value,
      naturalQuery: input("search-query").value,
      topK: Math.max(1, parseInt(input("search-k").value, 10) || 10),
      rerankBundle: select2("search-rerank").value
    });
    const hits = r.hits.map((h, i) => {
      const a = h.attribution;
      const signals = a ? Object.entries(a.signalScores).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(", ") : "";
      const spans = (a?.matchedSpans ?? []).map((m) => `${m.source}${m.label ? `:${m.label}` : ""} ${m.span?.start ?? 0}..${m.span?.end ?? 0}`);
      return el(
        "div",
        { class: "hit" },
        el("div", { class: "head" }, el("b", { text: `${i + 1}. ${h.docId}` }), el("span", { class: "tag", text: h.score.toFixed(4) }), el("span", { class: "muted", text: ` ${h.chunkId}` })),
        el("p", { text: h.text }),
        signals ? el("p", { class: "muted", text: `signals: ${signals}` }) : null,
        spans.length ? el("p", { class: "muted", text: `matched: ${spans.join("; ")}` }) : null
      );
    });
    out.replaceChildren(
      el("p", { class: "muted", text: `${r.hits.length} hits in ${r.resolvedIndex}; query terms: ${r.analyzedQueryTerms.join(" ") || "none"}` }),
      ...hits,
      r.documentHeat.length ? el("h3", { text: "Document heat" }) : "",
      heatView(r)
    );
  } catch (e) {
    out.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}
function stop2() {
  indexing?.abort();
  indexing = null;
}
function init3() {
  $("search-create").addEventListener("submit", createIndex);
  $("search-add").addEventListener("click", addDocuments);
  $("search-form").addEventListener("submit", runSearch);
  $("search-refresh").addEventListener("click", list);
  onCapabilities((c) => {
    caps = c;
    const embed = c.bundles.filter((b) => b.task === "TASK_EMBED");
    options(select2("search-create-bundle"), embed.map((b) => b.name), embed.map((b) => `${b.name} (dim ${b.widths[0] ?? "?"})`));
    const rerank = c.bundles.filter((b) => b.task === "TASK_RERANK").map((b) => b.name);
    options(select2("search-rerank"), ["", ...rerank], ["no rerank", ...rerank]);
  });
  capabilities().catch(() => void 0);
  void list();
}

// demo/web/src/tabs/server.ts
async function status() {
  const s = $("server");
  try {
    const [live, ready, meta] = await Promise.all([inference.serverLive({}), inference.serverReady({}), inference.serverMetadata({})]);
    const state = ready.ready ? "ready" : live.live ? "live, still loading" : "not live";
    s.textContent = `${meta.name || "server"} ${meta.version}: ${state}`;
    s.className = "muted";
  } catch (e) {
    s.textContent = `The server is not reachable through the demo: ${describe(e)}`;
    s.className = "error";
  }
}
function renderCapabilities(c) {
  const box = $("server-caps");
  const bundles2 = table(
    ["name", "task", "model", "widths", "context", "batch", "roles", ""],
    c.bundles.map((b) => [
      b.name,
      b.task,
      b.model ? `${b.model.id}@${b.model.revision}` : "",
      b.widths.join(", "),
      String(b.contextLimit),
      String(b.batchLimit),
      b.roles.join(" "),
      unloadButton(b.name)
    ])
  );
  const profiles = table(
    ["profile", "version", "steps", "layers", "needs"],
    c.profiles.map((p) => [p.name, p.version, p.steps.join(" \u2192 "), p.layers.join(", "), p.requiredBundles.join(", ")])
  );
  const l = c.server;
  const limits = l ? el("p", { class: "muted", text: `At most ${l.maxLoadedModels} served models and ${l.maxSessionsPerModel} sessions per model.` }) : "";
  box.replaceChildren(el("h3", { text: "Bundles" }), bundles2, limits, el("h3", { text: "Analysis profiles" }), profiles);
}
function unloadButton(name) {
  const b = el("button", { type: "button", text: "Unload" });
  b.addEventListener("click", async () => {
    const note = $("server-load-note");
    try {
      await analysis.unloadBundle({ name });
      say(note, `Unloaded ${name}.`);
      await reload();
    } catch (e) {
      say(note, describe(e), "error");
    }
  });
  return b;
}
async function reload() {
  try {
    renderCapabilities(await refresh());
  } catch (e) {
    $("server-caps").replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}
async function load(ev) {
  ev.preventDefault();
  const note = $("server-load-note");
  say(note, "Loading\u2026", "busy");
  try {
    const r = await analysis.loadBundle({
      name: input("server-load-name").value.trim(),
      bundle: input("server-load-path").value.trim(),
      sessions: Math.max(1, parseInt(input("server-load-sessions").value, 10) || 1),
      selectDevice: input("server-load-select").checked
    });
    say(note, `Loaded ${r.capabilities?.name ?? ""} (${r.capabilities?.task ?? ""}).`);
    await reload();
  } catch (e) {
    say(note, describe(e), "error");
  }
}
async function loadCatalog() {
  const box = $("server-catalog");
  say($("server-catalog-note"), "Reading the catalog\u2026", "busy");
  try {
    const [cat, installed] = await Promise.all([catalog.listModelCatalog({}), catalog.listInstalledModels({})]);
    box.replaceChildren(
      el("h3", { text: "Catalog" }),
      table(
        ["id", "kind", "language", "version", "licence", "size", "installed"],
        cat.entries.map((e) => [e.id, e.kind, e.language, e.version, e.licence, size(e.totalBytes), e.installed ? "yes" : ""])
      ),
      el("h3", { text: "Installed" }),
      table(["id", "version", "files", "size"], installed.models.map((m) => [m.id, m.version, String(m.files.length), size(m.totalBytes)]))
    );
    say($("server-catalog-note"), "");
  } catch (e) {
    say($("server-catalog-note"), describe(e), "error");
  }
}
function init4() {
  const t = input("token");
  t.value = token();
  t.addEventListener("change", () => {
    setToken(t.value.trim());
    void status();
    void reload();
  });
  $("server-load").addEventListener("submit", load);
  $("server-catalog-load").addEventListener("click", loadCatalog);
  void status();
  void reload();
}

// demo/web/src/sha256.ts
var K = new Uint32Array([
  1116352408,
  1899447441,
  3049323471,
  3921009573,
  961987163,
  1508970993,
  2453635748,
  2870763221,
  3624381080,
  310598401,
  607225278,
  1426881987,
  1925078388,
  2162078206,
  2614888103,
  3248222580,
  3835390401,
  4022224774,
  264347078,
  604807628,
  770255983,
  1249150122,
  1555081692,
  1996064986,
  2554220882,
  2821834349,
  2952996808,
  3210313671,
  3336571891,
  3584528711,
  113926993,
  338241895,
  666307205,
  773529912,
  1294757372,
  1396182291,
  1695183700,
  1986661051,
  2177026350,
  2456956037,
  2730485921,
  2820302411,
  3259730800,
  3345764771,
  3516065817,
  3600352804,
  4094571909,
  275423344,
  430227734,
  506948616,
  659060556,
  883997877,
  958139571,
  1322822218,
  1537002063,
  1747873779,
  1955562222,
  2024104815,
  2227730452,
  2361852424,
  2428436474,
  2756734187,
  3204031479,
  3329325298
]);
var Sha256 = class {
  h = new Uint32Array([
    1779033703,
    3144134277,
    1013904242,
    2773480762,
    1359893119,
    2600822924,
    528734635,
    1541459225
  ]);
  block = new Uint8Array(64);
  filled = 0;
  total = 0;
  w = new Uint32Array(64);
  update(data) {
    let i = 0;
    this.total += data.length;
    if (this.filled) {
      const take = Math.min(64 - this.filled, data.length);
      this.block.set(data.subarray(0, take), this.filled);
      this.filled += take;
      i = take;
      if (this.filled < 64) return this;
      this.compress(this.block, 0);
      this.filled = 0;
    }
    for (; i + 64 <= data.length; i += 64) this.compress(data, i);
    if (i < data.length) {
      this.block.set(data.subarray(i));
      this.filled = data.length - i;
    }
    return this;
  }
  /** The digest as lowercase hex. The hasher is spent. */
  hex() {
    const bits = this.total * 8;
    const pad = new Uint8Array((this.filled < 56 ? 56 : 120) - this.filled + 8);
    pad[0] = 128;
    const dv = new DataView(pad.buffer);
    dv.setUint32(pad.length - 8, Math.floor(bits / 4294967296), false);
    dv.setUint32(pad.length - 4, bits >>> 0, false);
    this.update(pad);
    let out = "";
    for (const v of this.h) out += v.toString(16).padStart(8, "0");
    return out;
  }
  compress(b, at) {
    const w = this.w;
    for (let t = 0; t < 16; t++) {
      const j = at + t * 4;
      w[t] = b[j] << 24 | b[j + 1] << 16 | b[j + 2] << 8 | b[j + 3];
    }
    for (let t = 16; t < 64; t++) {
      const x = w[t - 15];
      const y = w[t - 2];
      const s0 = (x >>> 7 | x << 25) ^ (x >>> 18 | x << 14) ^ x >>> 3;
      const s1 = (y >>> 17 | y << 15) ^ (y >>> 19 | y << 13) ^ y >>> 10;
      w[t] = w[t - 16] + s0 + w[t - 7] + s1 | 0;
    }
    const h = this.h;
    let [a, bb, c, d, e, f, g, hh] = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]];
    for (let t = 0; t < 64; t++) {
      const S1 = (e >>> 6 | e << 26) ^ (e >>> 11 | e << 21) ^ (e >>> 25 | e << 7);
      const ch = e & f ^ ~e & g;
      const t1 = hh + S1 + ch + K[t] + w[t] | 0;
      const S0 = (a >>> 2 | a << 30) ^ (a >>> 13 | a << 19) ^ (a >>> 22 | a << 10);
      const maj = a & bb ^ a & c ^ bb & c;
      const t2 = S0 + maj | 0;
      hh = g;
      g = f;
      f = e;
      e = d + t1 | 0;
      d = c;
      c = bb;
      bb = a;
      a = t1 + t2 | 0;
    }
    h[0] += a;
    h[1] += bb;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }
};
async function sha256Blob(b, progress) {
  const s = new Sha256();
  const reader = b.stream().getReader();
  let read = 0;
  for (; ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    s.update(value);
    read += value.length;
    progress?.(read);
  }
  return s.hex();
}

// demo/web/src/tabs/vocabulary.ts
var busy = null;
var select3 = (id) => $(id);
function start() {
  busy?.abort();
  busy = new AbortController();
  return busy;
}
function stop3() {
  busy?.abort();
  busy = null;
}
function actions(id) {
  const get = el("button", { type: "button", text: "Download" });
  get.addEventListener("click", () => void fetchArtifact(id));
  const del = el("button", { type: "button", text: "Delete" });
  del.addEventListener("click", async () => {
    try {
      await vocabulary.deleteVocabularyArtifact({ artifactId: id });
      await list2();
    } catch (e) {
      say($("vocab-note"), describe(e), "error");
    }
  });
  return el("span", {}, get, " ", del);
}
async function list2() {
  const box = $("vocab-lists");
  try {
    const [f, d, v] = await Promise.all([vocabulary.listDictionaryFormats({}), vocabulary.listDictionaries({}), vocabulary.listVocabularies({})]);
    const importable = f.formats.filter((x) => x.importable);
    options(select3("vocab-format"), importable.map((x) => x.formatId), importable.map((x) => `${x.formatId}: ${x.description}`));
    options(select3("vocab-seed"), ["", ...d.dictionaries.map((x) => x.artifactId)], ["no seed dictionary", ...d.dictionaries.map((x) => x.name || x.artifactId)]);
    const l = f.limits;
    box.replaceChildren(
      l ? el("p", {
        class: "muted",
        text: l.store ? `Store on${l.admin ? ", writable" : ", read only"}. Dictionaries up to ${size(l.maxDictionaryUploadBytes)}, learning up to ${size(l.maxLearnBytes)}, n-grams up to ${l.maxNgramSize}.` : "The server keeps no vocabulary store."
      }) : "",
      el("h3", { text: "Dictionaries" }),
      table(
        ["name", "format", "language", "entries", "size", "artifact", ""],
        d.dictionaries.map((x) => [x.name, x.formatId, x.language, String(x.entryCount), size(x.byteSize), x.artifactId, actions(x.artifactId)])
      ),
      el("h3", { text: "Vocabularies" }),
      table(
        ["artifact", "language", "documents", "tokens", "terms", "n-grams", "size", ""],
        v.vocabularies.map((x) => [x.artifactId, x.language, String(x.documents), String(x.tokens), String(x.termCount), String(x.ngramCount), size(x.byteSize), actions(x.artifactId)])
      )
    );
  } catch (e) {
    box.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}
async function importDictionary(ev) {
  ev.preventDefault();
  const note = $("vocab-note");
  const file = input("vocab-file").files?.[0];
  if (!file) {
    say(note, "Pick a dictionary file.", "error");
    return;
  }
  const ac = start();
  try {
    const sha = await sha256Blob(file, (done) => say(note, `Hashing ${size(done)} of ${size(file.size)}\u2026`, "busy"));
    say(note, `Uploading ${size(file.size)}\u2026`, "busy");
    const d = await put(
      TurboVocabularyService.method.importDictionary,
      importBody(
        {
          formatId: select3("vocab-format").value,
          language: input("vocab-language").value.trim(),
          name: input("vocab-name").value.trim() || file.name,
          byteSize: BigInt(file.size),
          sha256: sha,
          hasHeaderRow: input("vocab-header").checked
        },
        file
      ),
      ac.signal
    );
    say(note, `Imported ${d.name || d.artifactId}: ${d.entryCount} entries.`);
    await list2();
  } catch (e) {
    say(note, describe(e), "error");
  }
}
async function learn(ev) {
  ev.preventDefault();
  const note = $("vocab-note");
  const typed = document.getElementById("vocab-docs").value.split("\n\n").map((t) => t.trim()).filter((t) => t.length);
  const docs = [...typed, ...[...input("vocab-learn-files").files ?? []]];
  if (!docs.length) {
    say(note, "Type documents (a blank line between) or pick files.", "error");
    return;
  }
  const ac = start();
  try {
    const body2 = learnBody(
      {
        minTermFrequency: parseInt(input("vocab-min-tf").value, 10) || 1,
        maxTerms: parseInt(input("vocab-max-terms").value, 10) || 0,
        lowercase: input("vocab-lowercase").checked,
        filterStopwords: input("vocab-stopwords").checked,
        includeNgrams: input("vocab-ngrams").checked,
        minNgramSize: 2,
        maxNgramSize: 3,
        dictionaryArtifactId: select3("vocab-seed").value
      },
      docs
    );
    say(note, `Uploading ${docs.length} documents, ${size(body2.size)}\u2026`, "busy");
    const v = await put(TurboVocabularyService.method.learnVocabulary, body2, ac.signal);
    say(note, `Learned ${v.artifactId}: ${v.termCount} terms and ${v.ngramCount} n-grams from ${v.documents} documents.`);
    await list2();
  } catch (e) {
    say(note, describe(e), "error");
  }
}
async function fetchArtifact(id) {
  const note = $("vocab-note");
  const ac = start();
  const parts = [];
  let name = id;
  let got = 0;
  try {
    for await (const c of vocabulary.downloadVocabulary({ artifactId: id }, { signal: ac.signal })) {
      const f = c.frame;
      if (f.case === "dictionary") name = `${f.value.name || id}.dictionary`;
      else if (f.case === "vocabulary") name = `${id}.vocabulary`;
      else if (f.case === "data") {
        parts.push(f.value);
        got += f.value.length;
        say(note, `Downloading ${name}: ${size(got)}\u2026`, "busy");
      }
    }
    download(name, parts, "application/octet-stream");
    say(note, `Downloaded ${name}, ${size(got)}.`);
  } catch (e) {
    say(note, describe(e), "error");
  }
}
function init5() {
  $("vocab-import").addEventListener("submit", importDictionary);
  $("vocab-learn").addEventListener("submit", learn);
  $("vocab-refresh").addEventListener("click", list2);
  $("vocab-stop").addEventListener("click", stop3);
  void list2();
}

// demo/web/src/main.ts
var stops = { analyze: stop, search: stop2, vocabulary: stop3 };
function show(name) {
  for (const [tab, stop4] of Object.entries(stops)) if (tab !== name) stop4();
  for (const b of document.querySelectorAll("nav.tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === name));
  for (const p of document.querySelectorAll("section.panel")) p.hidden = p.id !== `tab-${name}`;
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
}
function start2() {
  for (const b of document.querySelectorAll("nav.tabs button")) b.addEventListener("click", () => show(b.dataset.tab ?? "embed"));
  init4();
  init2();
  init();
  init3();
  init5();
  const want = location.hash.slice(1);
  show(document.getElementById(`tab-${want}`) ? want : "embed");
}
start2();
