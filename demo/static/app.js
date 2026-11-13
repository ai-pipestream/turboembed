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
  const result2 = uInt64ToString(bits.lo, bits.hi);
  return negative ? "-" + result2 : result2;
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
  let result2 = b & 127;
  b = this.buf[this.pos++];
  result2 |= (b & 127) << 7;
  if ((b & 128) === 0) {
    this.assertBounds();
    return result2;
  }
  b = this.buf[this.pos++];
  result2 |= (b & 127) << 14;
  if ((b & 128) === 0) {
    this.assertBounds();
    return result2;
  }
  b = this.buf[this.pos++];
  result2 |= (b & 127) << 21;
  if ((b & 128) === 0) {
    this.assertBounds();
    return result2;
  }
  b = this.buf[this.pos++];
  result2 |= (b & 15) << 28;
  for (let readBytes = 5; (b & 128) !== 0 && readBytes < 10; readBytes++)
    b = this.buf[this.pos++];
  if ((b & 128) !== 0)
    throw new Error("invalid varint");
  this.assertBounds();
  return result2 >>> 0;
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
function unsafeIsSet(target, field3) {
  const name = field3.localName;
  if (field3.oneof) {
    return target[field3.oneof.localName].case === name;
  }
  if (field3.presence != IMPLICIT) {
    return target[name] !== void 0 && Object.prototype.hasOwnProperty.call(target, name);
  }
  switch (field3.fieldKind) {
    case "list":
      return target[name].length > 0;
    case "map":
      return Object.keys(target[name]).length > 0;
    case "scalar":
      return !isScalarZeroValue(field3.scalar, target[name]);
    case "enum":
      return target[name] !== field3.enum.values[0].number;
  }
  throw new Error("message field with implicit presence");
}
function unsafeIsSetExplicit(target, localName) {
  return Object.prototype.hasOwnProperty.call(target, localName) && target[localName] !== void 0;
}
function unsafeGet(target, field3) {
  if (field3.oneof) {
    const oneof = target[field3.oneof.localName];
    if (oneof.case === field3.localName) {
      return oneof.value;
    }
    return void 0;
  }
  return target[field3.localName];
}
function unsafeSet(target, field3, value) {
  if (field3.oneof) {
    target[field3.oneof.localName] = {
      case: field3.localName,
      value
    };
  } else {
    target[field3.localName] = value;
  }
}
function setOwn(obj, key3, value) {
  if (key3 === "__proto__") {
    Object.defineProperty(obj, "__proto__", {
      value,
      writable: true,
      enumerable: true,
      configurable: true
    });
  } else {
    obj[key3] = value;
  }
}
function getOwn(obj, key3) {
  return Object.prototype.hasOwnProperty.call(obj, key3) ? obj[key3] : void 0;
}
function unsafeClear(target, field3) {
  const name = field3.localName;
  if (field3.oneof) {
    const oneofLocalName = field3.oneof.localName;
    if (target[oneofLocalName].case === name) {
      target[oneofLocalName] = { case: void 0 };
    }
  } else if (field3.presence != IMPLICIT) {
    delete target[name];
  } else {
    switch (field3.fieldKind) {
      case "map":
        target[name] = {};
        break;
      case "list":
        target[name] = [];
        break;
      case "enum":
        target[name] = field3.enum.values[0].number;
        break;
      case "scalar":
        target[name] = scalarZeroValue(field3.scalar, field3.longAsString);
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
function unescapeBytesDefaultValue(str2) {
  const b = [];
  const input2 = {
    tail: str2,
    c: "",
    next() {
      if (this.tail.length == 0) {
        return false;
      }
      this.c = this.tail[0];
      this.tail = this.tail.substring(1);
      return true;
    },
    take(n2) {
      if (this.tail.length >= n2) {
        const r = this.tail.substring(0, n2);
        this.tail = this.tail.substring(n2);
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
              const s2 = input2.c;
              const t = input2.take(2);
              if (t === false) {
                return false;
              }
              const n2 = parseInt(s2 + t, 8);
              if (Number.isNaN(n2)) {
                return false;
              }
              b.push(n2);
              break;
            }
            case "x": {
              const s2 = input2.c;
              const t = input2.take(2);
              if (t === false) {
                return false;
              }
              const n2 = parseInt(s2 + t, 16);
              if (Number.isNaN(n2)) {
                return false;
              }
              b.push(n2);
              break;
            }
            case "u": {
              const s2 = input2.c;
              const t = input2.take(4);
              if (t === false) {
                return false;
              }
              const n2 = parseInt(s2 + t, 16);
              if (Number.isNaN(n2)) {
                return false;
              }
              const chunk = new Uint8Array(4);
              const view = new DataView(chunk.buffer);
              view.setInt32(0, n2, true);
              b.push(chunk[0], chunk[1], chunk[2], chunk[3]);
              break;
            }
            case "U": {
              const s2 = input2.c;
              const t = input2.take(8);
              if (t === false) {
                return false;
              }
              const tc = protoInt64.uEnc(s2 + t);
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
    const result2 = this.buffer.slice(0, this.pos);
    this.pos = 0;
    this.stackPos = [];
    return result2;
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
    const start2 = this.pos;
    const { written } = this.encodeUtf8Into(value, buf.subarray(start2 + lenPrefixSizeGuess));
    const lenPrefixSize = varint32Size(written);
    if (lenPrefixSize != lenPrefixSizeGuess) {
      buf.copyWithin(start2 + lenPrefixSize, start2 + lenPrefixSizeGuess, start2 + lenPrefixSizeGuess + written);
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
    const start2 = this.pos;
    const tag = this.uint32();
    const bytesRead = this.pos - start2;
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
    let start2 = this.pos;
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
    return this.buf.subarray(start2, this.pos);
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
    let s2 = -(lo & 1);
    lo = (lo >>> 1 | (hi & 1) << 31) ^ s2;
    hi = hi >>> 1 ^ s2;
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
    let len = this.uint32(), start2 = this.pos;
    this.pos += len;
    this.assertBounds();
    return this.buf.subarray(start2, start2 + len);
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
    getFile(fileName2) {
      return files.get(fileName2);
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
    const field3 = newField(proto, message, reg, oneof, mapEntries);
    message.fields.push(field3);
    message.field[field3.localName] = field3;
    if (oneof === void 0) {
      message.members.push(field3);
    } else {
      oneof.fields.push(field3);
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
  const output2 = reg.getMessage(trimLeadingDot(proto.outputType));
  assert(input2, `invalid MethodDescriptorProto: input_type ${proto.inputType} not found`);
  assert(output2, `invalid MethodDescriptorProto: output_type ${proto.inputType} not found`);
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
    output: output2,
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
  const field3 = {
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
    field3.kind = "extension";
    field3.file = file;
    field3.parent = parent;
    field3.oneof = void 0;
    field3.typeName = typeName;
    field3.jsonName = `[${typeName}]`;
    toStr = () => `extension ${typeName}`;
    const extendee = reg.getMessage(trimLeadingDot(proto.extendee));
    assert(extendee, `invalid FieldDescriptorProto: extendee ${proto.extendee} not found`);
    field3.extendee = extendee;
  } else {
    const parent = parentOrFile;
    assert(parent.kind == "message");
    field3.parent = parent;
    field3.oneof = oneof;
    field3.localName = oneof ? protoCamelCase(proto.name) : safeObjectProperty(protoCamelCase(proto.name));
    field3.jsonName = proto.jsonName;
    toStr = () => `field ${parent.typeName}.${proto.name}`;
  }
  Object.defineProperty(field3, "toString", {
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
      field3.fieldKind = "map";
      const { key: key3, value } = findMapEntryFields(mapEntry);
      field3.mapKey = key3.scalar;
      field3.mapKind = value.fieldKind;
      field3.message = value.message;
      field3.delimitedEncoding = false;
      field3.enum = value.enum;
      field3.scalar = value.scalar;
      return field3;
    }
    field3.fieldKind = "list";
    switch (type) {
      case TYPE_MESSAGE:
      case TYPE_GROUP:
        field3.listKind = "message";
        field3.message = reg.getMessage(trimLeadingDot(proto.typeName));
        assert(field3.message);
        field3.delimitedEncoding = isDelimitedEncoding(proto, parentOrFile);
        break;
      case TYPE_ENUM:
        field3.listKind = "enum";
        field3.enum = reg.getEnum(trimLeadingDot(proto.typeName));
        assert(field3.enum);
        break;
      default:
        field3.listKind = "scalar";
        field3.scalar = type;
        field3.longAsString = jstype == JS_STRING;
        break;
    }
    field3.packed = isPackedField(proto, parentOrFile);
    return field3;
  }
  switch (type) {
    case TYPE_MESSAGE:
    case TYPE_GROUP:
      field3.fieldKind = "message";
      field3.message = reg.getMessage(trimLeadingDot(proto.typeName));
      assert(field3.message, `invalid FieldDescriptorProto: type_name ${proto.typeName} not found`);
      field3.delimitedEncoding = isDelimitedEncoding(proto, parentOrFile);
      field3.getDefaultValue = () => void 0;
      break;
    case TYPE_ENUM: {
      const enumeration = reg.getEnum(trimLeadingDot(proto.typeName));
      assert(enumeration !== void 0, `invalid FieldDescriptorProto: type_name ${proto.typeName} not found`);
      field3.fieldKind = "enum";
      field3.enum = reg.getEnum(trimLeadingDot(proto.typeName));
      field3.getDefaultValue = () => {
        return unsafeIsSetExplicit(proto, "defaultValue") ? parseTextFormatEnumValue(enumeration, proto.defaultValue) : void 0;
      };
      break;
    }
    default: {
      field3.fieldKind = "scalar";
      field3.scalar = type;
      field3.longAsString = jstype == JS_STRING;
      field3.getDefaultValue = () => {
        return unsafeIsSetExplicit(proto, "defaultValue") ? parseTextFormatScalarValue(type, proto.defaultValue) : void 0;
      };
      break;
    }
  }
  return field3;
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
function findEnumSharedPrefix(enumName2, values) {
  const prefix2 = camelToSnakeCase(enumName2) + "_";
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
  const key3 = mapEntry.fields.find((f) => f.number === 1);
  const value = mapEntry.fields.find((f) => f.number === 2);
  assert(key3 && key3.fieldKind == "scalar" && key3.scalar != ScalarType.BYTES && key3.scalar != ScalarType.FLOAT && key3.scalar != ScalarType.DOUBLE && value && value.fieldKind != "list" && value.fieldKind != "map");
  return { key: key3, value };
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
function bootFileDescriptorProto(init16) {
  const proto = /* @__PURE__ */ Object.create({
    syntax: "",
    edition: 0
  });
  return Object.assign(proto, Object.assign(Object.assign({ $typeName: "google.protobuf.FileDescriptorProto", dependency: [], publicDependency: [], weakDependency: [], optionDependency: [], service: [], extension: [] }, init16), { messageType: init16.messageType.map(bootDescriptorProto), enumType: init16.enumType.map(bootEnumDescriptorProto) }));
}
function bootDescriptorProto(init16) {
  var _a, _b, _c, _d, _e, _f, _g, _h;
  const proto = /* @__PURE__ */ Object.create({
    visibility: 0
  });
  return Object.assign(proto, {
    $typeName: "google.protobuf.DescriptorProto",
    name: init16.name,
    field: (_b = (_a = init16.field) === null || _a === void 0 ? void 0 : _a.map(bootFieldDescriptorProto)) !== null && _b !== void 0 ? _b : [],
    extension: [],
    nestedType: (_d = (_c = init16.nestedType) === null || _c === void 0 ? void 0 : _c.map(bootDescriptorProto)) !== null && _d !== void 0 ? _d : [],
    enumType: (_f = (_e = init16.enumType) === null || _e === void 0 ? void 0 : _e.map(bootEnumDescriptorProto)) !== null && _f !== void 0 ? _f : [],
    extensionRange: (_h = (_g = init16.extensionRange) === null || _g === void 0 ? void 0 : _g.map((e) => Object.assign({ $typeName: "google.protobuf.DescriptorProto.ExtensionRange" }, e))) !== null && _h !== void 0 ? _h : [],
    oneofDecl: [],
    reservedRange: [],
    reservedName: []
  });
}
function bootFieldDescriptorProto(init16) {
  const proto = /* @__PURE__ */ Object.create({
    label: 1,
    typeName: "",
    extendee: "",
    defaultValue: "",
    oneofIndex: 0,
    jsonName: "",
    proto3Optional: false
  });
  return Object.assign(proto, Object.assign(Object.assign({ $typeName: "google.protobuf.FieldDescriptorProto" }, init16), { options: init16.options ? bootFieldOptions(init16.options) : void 0 }));
}
function bootFieldOptions(init16) {
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
  return Object.assign(proto, Object.assign(Object.assign({ $typeName: "google.protobuf.FieldOptions" }, init16), { targets: (_a = init16.targets) !== null && _a !== void 0 ? _a : [], editionDefaults: (_c = (_b = init16.editionDefaults) === null || _b === void 0 ? void 0 : _b.map((e) => Object.assign({ $typeName: "google.protobuf.FieldOptions.EditionDefault" }, e))) !== null && _c !== void 0 ? _c : [], uninterpretedOption: [] }));
}
function bootEnumDescriptorProto(init16) {
  const proto = /* @__PURE__ */ Object.create({
    visibility: 0
  });
  return Object.assign(proto, {
    $typeName: "google.protobuf.EnumDescriptorProto",
    name: init16.name,
    reservedName: [],
    reservedRange: [],
    value: init16.value.map((e) => Object.assign({ $typeName: "google.protobuf.EnumValueDescriptorProto" }, e))
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
      const result2 = nativeSetFromBase64.call(bytes2, base64Str);
      if (result2.read == len) {
        written = result2.written;
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
function isReflectList(arg, field3) {
  var _a, _b, _c, _d;
  if (isObject(arg) && unsafeLocal in arg && "add" in arg && "field" in arg && typeof arg.field == "function") {
    if (field3 !== void 0) {
      const a = field3;
      const b = arg.field();
      return a.listKind == b.listKind && a.scalar === b.scalar && ((_a = a.message) === null || _a === void 0 ? void 0 : _a.typeName) === ((_b = b.message) === null || _b === void 0 ? void 0 : _b.typeName) && ((_c = a.enum) === null || _c === void 0 ? void 0 : _c.typeName) === ((_d = b.enum) === null || _d === void 0 ? void 0 : _d.typeName);
    }
    return true;
  }
  return false;
}
function isReflectMap(arg, field3) {
  var _a, _b, _c, _d;
  if (isObject(arg) && unsafeLocal in arg && "has" in arg && "field" in arg && typeof arg.field == "function") {
    if (field3 !== void 0) {
      const a = field3, b = arg.field();
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
function create(schema, init16) {
  if (isMessage(init16, schema)) {
    return init16;
  }
  return compiledCreate(schema)(init16);
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
  return (init16) => {
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
      const initValue = init16 === null || init16 === void 0 ? void 0 : init16[name];
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
  for (const field3 of oneof.fields) {
    let convert;
    if (field3.fieldKind == "message") {
      convert = compileConvertMessage(field3);
    } else if (field3.fieldKind == "scalar" && field3.scalar == ScalarType.BYTES) {
      convert = toU8Arr;
    }
    converters.set(field3.localName, convert !== null && convert !== void 0 ? convert : ((value) => value));
  }
  return converters;
}
function compileConvertMessage(field3) {
  if (field3.fieldKind == "message" && !field3.oneof && isWrapperDesc(field3.message)) {
    return field3.message.fields[0].scalar == ScalarType.BYTES ? toU8Arr : void 0;
  }
  if (field3.message.typeName == "google.protobuf.Struct" && field3.parent.typeName !== "google.protobuf.Value") {
    return void 0;
  }
  const messageDesc2 = field3.message;
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
function createZeroValue(field3) {
  const defaultValue = field3.getDefaultValue();
  if (defaultValue !== void 0) {
    return field3.fieldKind == "scalar" && field3.longAsString ? defaultValue.toString() : defaultValue;
  }
  return field3.fieldKind == "scalar" ? scalarZeroValue(field3.scalar, field3.longAsString) : field3.enum.values[0].number;
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/reflect/message.js
var NULL_VALUE = 0;
function localMessageMapper(field3) {
  if (usesJsonRepresentation(field3)) {
    return {
      toMessage: (local) => wktStructToReflect(local),
      toLocal: (message) => wktStructToLocal(message)
    };
  }
  if (field3.fieldKind == "message" && !field3.oneof && isWrapperDesc(field3.message)) {
    const wrapperDesc = field3.message;
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
  const childDesc = field3.message;
  return {
    toMessage: (local) => local === void 0 ? create(childDesc) : local,
    toLocal: (message) => message
  };
}
function usesJsonRepresentation(field3) {
  return field3.message.typeName == "google.protobuf.Struct" && field3.parent.typeName != "google.protobuf.Value";
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
  for (const field3 of sortedFields) {
    fieldWriters.push(compileField(field3));
  }
  return compiled;
}
function compileField(field3) {
  switch (field3.fieldKind) {
    case "message":
    case "scalar":
    case "enum":
      return compileSingularField(field3);
    case "list":
      return compileListField(field3);
    case "map":
      return compileMapField(field3);
  }
}
function compileSingularField(field3) {
  const writeValue = compileSingularValue(field3);
  const localName = field3.localName;
  if (field3.oneof) {
    const oneofLocalName = field3.oneof.localName;
    return (writer, opts, message) => {
      const oneof = message[oneofLocalName];
      if (oneof.case === localName) {
        writeValue(writer, opts, oneof.value);
      }
    };
  }
  if (field3.presence != IMPLICIT4) {
    const requiredError = field3.presence == LEGACY_REQUIRED2 ? `cannot encode ${field3} to binary: required field not set` : void 0;
    return (writer, opts, message) => {
      const value = message[localName];
      if (value !== void 0 && Object.prototype.hasOwnProperty.call(message, localName)) {
        writeValue(writer, opts, value);
      } else if (requiredError !== void 0) {
        throw new Error(requiredError);
      }
    };
  }
  if (field3.fieldKind == "enum") {
    const zero = field3.enum.values[0].number;
    return (writer, opts, message) => {
      const value = message[localName];
      if (value !== zero) {
        writeValue(writer, opts, value);
      }
    };
  }
  switch (field3.scalar) {
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
function compileSingularValue(field3) {
  switch (field3.fieldKind) {
    case "message": {
      const { toMessage } = localMessageMapper(field3);
      const writeChild = compileChildWriter(field3);
      return (writer, opts, value) => {
        writeChild(writer, opts, toMessage(value));
      };
    }
    case "scalar":
    case "enum": {
      const scalarType = field3.fieldKind == "enum" ? ScalarType.INT32 : field3.scalar;
      const fieldNo = field3.number;
      const wireType = writeTypeOfScalar(scalarType);
      const writeScalar = compileScalarValue(scalarType, field3.parent.typeName, field3.name);
      return (writer, opts, value) => {
        writer.tag(fieldNo, wireType);
        writeScalar(writer, value);
      };
    }
  }
}
function compileListField(field3) {
  const localName = field3.localName;
  const fieldNo = field3.number;
  switch (field3.listKind) {
    case "message": {
      const { toMessage } = localMessageMapper(field3);
      const writeChild = compileChildWriter(field3);
      return (writer, opts, message) => {
        const items = message[localName];
        for (let i = 0; i < items.length; i++) {
          writeChild(writer, opts, toMessage(items[i]));
        }
      };
    }
    case "scalar":
    case "enum": {
      const scalarType = field3.listKind == "enum" ? ScalarType.INT32 : field3.scalar;
      const writeScalar = compileScalarValue(scalarType, field3.parent.typeName, field3.name);
      if (field3.packed) {
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
function compileMapField(field3) {
  const localName = field3.localName;
  const fieldNo = field3.number;
  const writeKey = compileMapKey(field3);
  if (field3.mapKind == "message") {
    const { toMessage } = localMessageMapper(field3);
    const writeMessage = compiledWriter(field3.message);
    return (writer, opts, message) => {
      const record = message[localName];
      const keys = Object.keys(record);
      for (let i = 0; i < keys.length; i++) {
        const key3 = keys[i];
        writer.tag(fieldNo, WireType.LengthDelimited).fork();
        writeKey(writer, key3);
        writer.tag(2, WireType.LengthDelimited).fork();
        writeMessage(writer, opts, toMessage(record[key3]));
        writer.join();
        writer.join();
      }
    };
  }
  const scalarType = field3.mapKind == "enum" ? ScalarType.INT32 : field3.scalar;
  const valueWireType = writeTypeOfScalar(scalarType);
  const writeScalar = compileScalarValue(scalarType, field3.parent.typeName, field3.name);
  return (writer, opts, message) => {
    const record = message[localName];
    const keys = Object.keys(record);
    for (let i = 0; i < keys.length; i++) {
      const key3 = keys[i];
      writer.tag(fieldNo, WireType.LengthDelimited).fork();
      writeKey(writer, key3);
      writer.tag(2, valueWireType);
      writeScalar(writer, record[key3]);
      writer.join();
    }
  };
}
function compileMapKey(field3) {
  const wireType = writeTypeOfScalar(field3.mapKey);
  const writeScalar = compileScalarValue(field3.mapKey, field3.parent.typeName, field3.name);
  const convertKey = compileMapKeyConverter(field3.mapKey);
  return (writer, key3) => {
    writer.tag(1, wireType);
    writeScalar(writer, convertKey(key3));
  };
}
function compileMapKeyConverter(type) {
  switch (type) {
    case ScalarType.STRING:
      return (key3) => key3;
    case ScalarType.BOOL:
      return (key3) => key3 === "true" ? true : key3 === "false" ? false : key3;
    case ScalarType.UINT64:
    case ScalarType.FIXED64:
      return (key3) => {
        try {
          return protoInt64.uParse(key3);
        } catch (_a) {
          return key3;
        }
      };
    case ScalarType.INT64:
    case ScalarType.SFIXED64:
    case ScalarType.SINT64:
      return (key3) => {
        try {
          return protoInt64.parse(key3);
        } catch (_a) {
          return key3;
        }
      };
    default:
      return (key3) => {
        const n2 = Number.parseInt(key3);
        return Number.isFinite(n2) ? n2 : key3;
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
function writeField(writer, opts, msg, field3) {
  compileField(field3)(writer, opts, msg[unsafeLocal]);
}
function compileChildWriter(field3) {
  const fieldNo = field3.number;
  const writeMessage = compiledWriter(field3.message);
  if (field3.delimitedEncoding) {
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
function checkField(field3, value) {
  const check2 = field3.fieldKind == "list" ? isReflectList(value, field3) : field3.fieldKind == "map" ? isReflectMap(value, field3) : checkSingular(field3, value);
  if (check2 === true) {
    return void 0;
  }
  let reason;
  switch (field3.fieldKind) {
    case "list":
      reason = `expected ${formatReflectList(field3)}, got ${formatVal(value)}`;
      break;
    case "map":
      reason = `expected ${formatReflectMap(field3)}, got ${formatVal(value)}`;
      break;
    default: {
      reason = reasonSingular(field3, value, check2);
    }
  }
  return new FieldError(field3, reason);
}
function checkListItem(field3, index, value) {
  const check2 = checkSingular(field3, value);
  if (check2 !== true) {
    return new FieldError(field3, `list item #${index + 1}: ${reasonSingular(field3, value, check2)}`);
  }
  return void 0;
}
function checkMapEntry(field3, key3, value) {
  const checkKey = checkScalarValue(field3.mapKey)(key3);
  if (checkKey !== true) {
    return new FieldError(field3, `invalid map key: ${reasonSingular({ scalar: field3.mapKey }, key3, checkKey)}`);
  }
  const checkVal = checkSingular(field3, value);
  if (checkVal !== true) {
    return new FieldError(field3, `map entry ${formatVal(key3)}: ${reasonSingular(field3, value, checkVal)}`);
  }
  return void 0;
}
function checkSingular(field3, value) {
  if (field3.scalar !== void 0) {
    return checkScalarValue(field3.scalar)(value);
  }
  if (field3.enum !== void 0) {
    if (field3.enum.open) {
      return checkScalarValue(ScalarType.INT32)(value);
    }
    return field3.enum.values.some((v) => v.number === value);
  }
  return isReflectMessage(value, field3.message);
}
function checkScalarValue(scalar2) {
  switch (scalar2) {
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
function reasonSingular(field3, val, details) {
  details = typeof details == "string" ? `: ${details}` : `, got ${formatVal(val)}`;
  if (field3.scalar !== void 0) {
    return `expected ${scalarTypeDescription(field3.scalar)}` + details;
  }
  if (field3.enum !== void 0) {
    return `expected ${field3.enum.toString()}` + details;
  }
  return `expected ${formatReflectMessage(field3.message)}` + details;
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
function formatReflectList(field3) {
  switch (field3.listKind) {
    case "message":
      return `ReflectList (${field3.message.toString()})`;
    case "enum":
      return `ReflectList (${field3.enum.toString()})`;
    case "scalar":
      return `ReflectList (${ScalarType[field3.scalar]})`;
  }
}
function formatReflectMap(field3) {
  switch (field3.mapKind) {
    case "message":
      return `ReflectMap (${ScalarType[field3.mapKey]}, ${field3.message.toString()})`;
    case "enum":
      return `ReflectMap (${ScalarType[field3.mapKey]}, ${field3.enum.toString()})`;
    case "scalar":
      return `ReflectMap (${ScalarType[field3.mapKey]}, ${ScalarType[field3.scalar]})`;
  }
}
function scalarTypeDescription(scalar2) {
  switch (scalar2) {
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
function reflect(messageDesc2, message, check2 = true) {
  return new ReflectMessageImpl(messageDesc2, message, check2);
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
  constructor(messageDesc2, message, check2 = true) {
    this.lists = /* @__PURE__ */ new Map();
    this.maps = /* @__PURE__ */ new Map();
    this.check = check2;
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
  isSet(field3) {
    assertOwn(this.message, field3);
    return unsafeIsSet(this.message, field3);
  }
  clear(field3) {
    assertOwn(this.message, field3);
    unsafeClear(this.message, field3);
  }
  get(field3) {
    assertOwn(this.message, field3);
    const value = unsafeGet(this.message, field3);
    switch (field3.fieldKind) {
      case "list":
        let list5 = this.lists.get(field3);
        if (!list5 || list5[unsafeLocal] !== value) {
          this.lists.set(
            field3,
            // biome-ignore lint/suspicious/noAssignInExpressions: no
            list5 = new ReflectListImpl(field3, value, this.check)
          );
        }
        return list5;
      case "map":
        let map = this.maps.get(field3);
        if (!map || map[unsafeLocal] !== value) {
          this.maps.set(
            field3,
            // biome-ignore lint/suspicious/noAssignInExpressions: no
            map = new ReflectMapImpl(field3, value, this.check)
          );
        }
        return map;
      case "message":
        return messageToReflect(field3, value, this.check);
      case "scalar":
        return value === void 0 ? scalarZeroValue(field3.scalar, false) : longToReflect(field3, value);
      case "enum":
        return value !== null && value !== void 0 ? value : field3.enum.values[0].number;
    }
  }
  set(field3, value) {
    assertOwn(this.message, field3);
    if (this.check) {
      const err = checkField(field3, value);
      if (err) {
        throw err;
      }
    }
    let local;
    if (field3.fieldKind == "message") {
      local = messageToLocal(field3, value);
    } else if (isReflectMap(value) || isReflectList(value)) {
      local = value[unsafeLocal];
    } else {
      local = longToLocal(field3, value);
    }
    unsafeSet(this.message, field3, local);
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
  constructor(field3, unsafeInput, check2) {
    this._field = field3;
    this._arr = this[unsafeLocal] = unsafeInput;
    this.check = check2;
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
  constructor(field3, unsafeInput, check2 = true) {
    this.obj = this[unsafeLocal] = unsafeInput !== null && unsafeInput !== void 0 ? unsafeInput : {};
    this.check = check2;
    this._field = field3;
  }
  field() {
    return this._field;
  }
  set(key3, value) {
    if (this.check) {
      const err = checkMapEntry(this._field, key3, value);
      if (err) {
        throw err;
      }
    }
    setOwn(this.obj, mapKeyToLocal(key3), mapValueToLocal(this._field, value));
    return this;
  }
  delete(key3) {
    const k = mapKeyToLocal(key3);
    const has = Object.prototype.hasOwnProperty.call(this.obj, k);
    if (has) {
      delete this.obj[k];
    }
    return has;
  }
  clear() {
    for (const key3 of Object.keys(this.obj)) {
      delete this.obj[key3];
    }
  }
  get(key3) {
    let val = getOwn(this.obj, mapKeyToLocal(key3));
    if (val !== void 0) {
      val = mapValueToReflect(this._field, val, this.check);
    }
    return val;
  }
  has(key3) {
    return Object.prototype.hasOwnProperty.call(this.obj, mapKeyToLocal(key3));
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
function messageToLocal(field3, value) {
  if (!isReflectMessage(value)) {
    return value;
  }
  if (isWrapper(value.message) && !field3.oneof && field3.fieldKind == "message") {
    return value.message.value;
  }
  if (value.desc.typeName == "google.protobuf.Struct" && field3.parent.typeName != "google.protobuf.Value") {
    return wktStructToLocal(value.message);
  }
  return value.message;
}
function messageToReflect(field3, value, check2) {
  if (value !== void 0) {
    if (isWrapperDesc(field3.message) && !field3.oneof && field3.fieldKind == "message") {
      value = {
        $typeName: field3.message.typeName,
        value: longToReflect(field3.message.fields[0], value)
      };
    } else if (field3.message.typeName == "google.protobuf.Struct" && field3.parent.typeName != "google.protobuf.Value" && isObject(value)) {
      value = wktStructToReflect(value);
    }
  }
  return new ReflectMessageImpl(field3.message, value, check2);
}
function listItemToLocal(field3, value) {
  if (field3.listKind == "message") {
    return messageToLocal(field3, value);
  }
  return longToLocal(field3, value);
}
function listItemToReflect(field3, value, check2) {
  if (field3.listKind == "message") {
    return messageToReflect(field3, value, check2);
  }
  return longToReflect(field3, value);
}
function mapValueToLocal(field3, value) {
  if (field3.mapKind == "message") {
    return messageToLocal(field3, value);
  }
  return longToLocal(field3, value);
}
function mapValueToReflect(field3, value, check2) {
  if (field3.mapKind == "message") {
    return messageToReflect(field3, value, check2);
  }
  return value;
}
function mapKeyToLocal(key3) {
  return typeof key3 == "string" || typeof key3 == "number" ? key3 : String(key3);
}
function mapKeyToReflect(key3, type) {
  switch (type) {
    case ScalarType.STRING:
      return key3;
    case ScalarType.INT32:
    case ScalarType.FIXED32:
    case ScalarType.UINT32:
    case ScalarType.SFIXED32:
    case ScalarType.SINT32: {
      const n2 = Number.parseInt(key3);
      if (Number.isFinite(n2)) {
        return n2;
      }
      break;
    }
    case ScalarType.BOOL:
      switch (key3) {
        case "true":
          return true;
        case "false":
          return false;
      }
      break;
    case ScalarType.UINT64:
    case ScalarType.FIXED64:
      try {
        return protoInt64.uParse(key3);
      } catch (_a) {
      }
      break;
    default:
      try {
        return protoInt64.parse(key3);
      } catch (_b) {
      }
      break;
  }
  return key3;
}
function longToReflect(field3, value) {
  switch (field3.scalar) {
    case ScalarType.INT64:
    case ScalarType.SFIXED64:
    case ScalarType.SINT64:
      if ("longAsString" in field3 && field3.longAsString && typeof value == "string") {
        value = protoInt64.parse(value);
      }
      break;
    case ScalarType.FIXED64:
    case ScalarType.UINT64:
      if ("longAsString" in field3 && field3.longAsString && typeof value == "string") {
        value = protoInt64.uParse(value);
      }
      break;
  }
  return value;
}
function longToLocal(field3, value) {
  switch (field3.scalar) {
    case ScalarType.INT64:
    case ScalarType.SFIXED64:
    case ScalarType.SINT64:
      if ("longAsString" in field3 && field3.longAsString) {
        value = String(value);
      } else if (typeof value == "string" || typeof value == "number") {
        value = protoInt64.parse(value);
      }
      break;
    case ScalarType.FIXED64:
    case ScalarType.UINT64:
      if ("longAsString" in field3 && field3.longAsString) {
        value = String(value);
      } else if (typeof value == "string" || typeof value == "number") {
        value = protoInt64.uParse(value);
      }
      break;
  }
  return value;
}

// demo/web/node_modules/@bufbuild/protobuf/dist/esm/codegenv2/message.js
function messageDesc(file, path2, ...paths) {
  return paths.reduce((acc, cur) => acc.nestedMessages[cur], file.messages[path2]);
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
  for (const field3 of desc.fields) {
    fieldReaders.set(field3.number, compileFieldReader(field3));
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
function readField(message, reader, field3, wireType, ctx) {
  compileFieldReader(field3)(message[unsafeLocal], reader, ctx, wireType);
}
function compileFieldReader(field3) {
  switch (field3.fieldKind) {
    case "scalar":
      return compileScalarFieldReader(field3);
    case "enum":
      return compileEnumFieldReader(field3);
    case "message":
      return compileMessageFieldReader(field3);
    case "list":
      return compileListFieldReader(field3);
    case "map":
      return compileMapFieldReader(field3);
  }
}
function compileScalarFieldReader(field3) {
  const readScalar = compileScalarReader(field3.scalar, field3.utf8Validation, field3.longAsString);
  const localName = field3.localName;
  if (field3.oneof) {
    const oneofLocalName = field3.oneof.localName;
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
function compileEnumFieldReader(field3) {
  var _a;
  const localName = field3.localName;
  const oneofLocalName = (_a = field3.oneof) === null || _a === void 0 ? void 0 : _a.localName;
  if (field3.enum.open) {
    if (oneofLocalName !== void 0) {
      return (message, reader) => {
        message[oneofLocalName] = { case: localName, value: reader.int32() };
      };
    }
    return (message, reader) => {
      message[localName] = reader.int32();
    };
  }
  const values = field3.enum.values;
  const fieldNo = field3.number;
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
function compileMessageFieldReader(field3) {
  const localName = field3.localName;
  const { toMessage, toLocal } = localMessageMapper(field3);
  const readChild = compileChildReader(field3);
  if (field3.oneof) {
    const oneofLocalName = field3.oneof.localName;
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
function compileChildReader(field3) {
  const compiledChild = compiledReader(field3.message);
  if (field3.delimitedEncoding) {
    const fieldNo = field3.number;
    return (child, reader, ctx) => compiledChild.readGroup(child, reader, ctx, fieldNo);
  }
  return (child, reader, ctx) => compiledChild.read(child, reader, ctx, reader.uint32());
}
function compileListFieldReader(field3) {
  const localName = field3.localName;
  if (field3.listKind == "message") {
    const { toMessage, toLocal } = localMessageMapper(field3);
    const readChild = compileChildReader(field3);
    return (message, reader, ctx) => {
      const child = toMessage(void 0);
      readChild(child, reader, ctx);
      message[localName].push(toLocal(child));
    };
  }
  const scalarType = field3.listKind == "enum" ? ScalarType.INT32 : field3.scalar;
  const longAsString = field3.listKind == "scalar" ? field3.longAsString : false;
  const readScalar = compileScalarReader(scalarType, field3.utf8Validation, longAsString);
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
function compileMapFieldReader(field3) {
  const localName = field3.localName;
  const readKey = compileScalarReader(field3.mapKey, field3.utf8Validation, false);
  const keyZero = scalarZeroValue(field3.mapKey, false);
  let readValue;
  let valueDefault;
  switch (field3.mapKind) {
    case "scalar": {
      const scalar2 = field3.scalar;
      const readScalar = compileScalarReader(scalar2, field3.utf8Validation, false);
      readValue = (reader) => readScalar(reader);
      if (scalar2 == ScalarType.BYTES) {
        valueDefault = () => new Uint8Array(0);
      } else {
        const zero = scalarZeroValue(scalar2, false);
        valueDefault = () => zero;
      }
      break;
    }
    case "enum": {
      const zero = field3.enum.values[0].number;
      readValue = (reader) => reader.int32();
      valueDefault = () => zero;
      break;
    }
    case "message": {
      const { toMessage, toLocal } = localMessageMapper(field3);
      const readChild = compiledReader(field3.message).read;
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
    let key3;
    let val;
    const len = reader.uint32();
    const end = reader.pos + len;
    while (reader.pos < end) {
      const [fieldNo] = reader.tag();
      switch (fieldNo) {
        case 1:
          key3 = readKey(reader);
          break;
        case 2:
          val = readValue(reader, ctx, val);
          break;
      }
    }
    if (key3 === void 0) {
      key3 = keyZero;
    }
    if (val === void 0) {
      val = valueDefault();
    }
    setOwn(record, key3, val);
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
function serviceDesc(file, path2, ...paths) {
  if (paths.length > 0) {
    throw new Error();
  }
  return file.services[path2];
}

// demo/web/src/gen/turbo_typed_annotations_pb.ts
var file_turbo_typed_annotations = /* @__PURE__ */ fileDesc("Ch10dXJib190eXBlZF9hbm5vdGF0aW9ucy5wcm90bxIPdHVyYm8ucmVzdWx0LnYxIjgKCE1vZGVsUmVmEgoKAmlkGAEgASgJEhAKCHJldmlzaW9uGAIgASgJEg4KBnNoYTI1NhgDIAEoCSKjAQoKTGF5ZXJFcnJvchItCgRraW5kGAEgASgOMh8udHVyYm8ucmVzdWx0LnYxLkxheWVyRXJyb3JLaW5kEhIKCnR1cmJvX2NvZGUYAiABKAUSNQoFbGF5ZXIYAyABKAsyIS50dXJiby5yZXN1bHQudjEuTGF5ZXJJbnN0YW5jZVJlZkgAiAEBEhEKCWFubm90YXRvchgEIAEoCUIICgZfbGF5ZXIiwQEKCFZpZXdJbmZvEhAKCGluc3RhbmNlGAEgASgJEjAKBnN0YXR1cxgCIAEoDjIgLnR1cmJvLnJlc3VsdC52MS5MYXllcldpcmVTdGF0dXMSEQoJYW5ub3RhdG9yGAQgASgJEg4KBnBhcmVudBgFIAEoCRIvCgVlcnJvchgGIAEoCzIbLnR1cmJvLnJlc3VsdC52MS5MYXllckVycm9ySACIAQFCCAoGX2Vycm9ySgQIAxAEUg1zdGF0dXNfZGV0YWlsIr4CChtJbnNwZWN0RG9jdW1lbnRHcmFwaFJlcXVlc3QSDgoGZG9jX2lkGAEgASgJEg0KBWdyYXBoGAIgASgMEgwKBHZpZXcYAyABKA0SLAoHb2Zmc2V0cxgEIAEoDjIbLnR1cmJvLnJlc3VsdC52MS5PZmZzZXRNb2RlEjMKB2JpbmRpbmcYBSABKAsyHS50dXJiby5yZXN1bHQudjEuSW5wdXRCaW5kaW5nSACIAQESFQoNb3BlcmF0aW9uX2tleRgGIAEoCRI1CgpwaWlfb3V0cHV0GAcgASgLMiEudHVyYm8ucmVzdWx0LnYxLlBpaU91dHB1dE9wdGlvbnMSNQoKcGlpX3BvbGljeRgIIAEoCzIhLnR1cmJvLnJlc3VsdC52MS5QaWlSZWxlYXNlUG9saWN5QgoKCF9iaW5kaW5nIj4KEFBpaVJlbGVhc2VQb2xpY3kSEAoGaW5saW5lGAEgASgMSAASDgoEbmFtZRgCIAEoCUgAQggKBnBvbGljeSJhCgxJbnB1dEJpbmRpbmcSFwoPc291cmNlX3JldmlzaW9uGAEgASgJEhkKEXBhcnNlX2FydGlmYWN0X2lkGAIgASgJEh0KFXBhcnNlX2FydGlmYWN0X2RpZ2VzdBgDIAEoDCK2AQoNSW5wdXRJZGVudGl0eRIzCgdiaW5kaW5nGAEgASgLMh0udHVyYm8ucmVzdWx0LnYxLklucHV0QmluZGluZ0gAiAEBEhgKEHJhd19ieXRlc19kaWdlc3QYAiABKAwSFQoNcmF3X2J5dGVzX2xlbhgDIAEoBBIaChJzb3VyY2VfdGV4dF9kaWdlc3QYBCABKAwSFwoPc291cmNlX3RleHRfbGVuGAUgASgNQgoKCF9iaW5kaW5nIpMBCg5SZXN1bHRJZGVudGl0eRIRCglyZXN1bHRfaWQYASABKAkSFgoOY29udGVudF9kaWdlc3QYAiABKAwSFwoPZ3JhcGhfYnl0ZXNfbGVuGAMgASgEEhwKFGdyYXBoX2Zvcm1hdF92ZXJzaW9uGAQgASgNEh8KF2VudmVsb3BlX3NjaGVtYV92ZXJzaW9uGAUgASgNIjMKEExheWVySW5zdGFuY2VSZWYSDQoFbGF5ZXIYASABKAkSEAoIaW5zdGFuY2UYAiABKAkixQIKD0xheWVyUHJvdmVuYW5jZRINCgVsYXllchgBIAEoCRIQCghpbnN0YW5jZRgCIAEoCRIRCgltb2RlbF9yZWYYAyABKAkSGQoRcHJvdmVuYW5jZV9kaWdlc3QYBCABKAwSGwoOY29udGVudF9kaWdlc3QYBSABKAxIAIgBARIiChVpbnB1dF9zbmFwc2hvdF9kaWdlc3QYBiABKAxIAYgBARIUCgxjb2RlY19zY2hlbWEYByABKA0SMAoFcmVhZHMYCCADKAsyIS50dXJiby5yZXN1bHQudjEuTGF5ZXJJbnN0YW5jZVJlZhItCgJvbhgJIAMoCzIhLnR1cmJvLnJlc3VsdC52MS5MYXllckluc3RhbmNlUmVmQhEKD19jb250ZW50X2RpZ2VzdEIYChZfaW5wdXRfc25hcHNob3RfZGlnZXN0In8KCVNjb3BlZFJlZhIRCglyZXN1bHRfaWQYASABKAkSDQoFbGF5ZXIYAiABKAkSEAoIaW5zdGFuY2UYAyABKAkSDQoFaW5kZXgYBCABKA0SFQoNdmlld19pbnN0YW5jZRgFIAEoCRIYChB2aWV3X3RleHRfZGlnZXN0GAYgASgMIm4KB0VkaXRSdW4SFAoMcGFyZW50X3N0YXJ0GAEgASgNEhQKDHBhcmVudF93aWR0aBgCIAEoDRITCgtjaGlsZF9zdGFydBgDIAEoDRITCgtjaGlsZF93aWR0aBgEIAEoDRINCgVjb3VudBgFIAEoDSKvAQoJVmlld0NoYWluEgwKBHZpZXcYASABKA0SEAoIaW5zdGFuY2UYAiABKAkSEwoGcGFyZW50GAMgASgNSACIAQESEwoLdGV4dF9kaWdlc3QYBCABKAwSEAoIdGV4dF9sZW4YBSABKA0SEgoKZWRpdF9jb3VudBgGIAEoDRInCgVlZGl0cxgHIAMoCzIYLnR1cmJvLnJlc3VsdC52MS5FZGl0UnVuQgkKB19wYXJlbnQiWgoNSW5saW5lTWFwcGluZxIeChZtYXBwaW5nX3NjaGVtYV92ZXJzaW9uGAEgASgNEikKBXZpZXdzGAIgAygLMhoudHVyYm8ucmVzdWx0LnYxLlZpZXdDaGFpbiJdCg5NYXBwaW5nUGF5bG9hZBIwCgZpbmxpbmUYASABKAsyHi50dXJiby5yZXN1bHQudjEuSW5saW5lTWFwcGluZ0gAQgkKB3BheWxvYWRKBAgCEANSCGFydGlmYWN0IicKCUJ5dGVSYW5nZRINCgVzdGFydBgBIAEoDRILCgNlbmQYAiABKA0iKgoLRGV0ZWN0b3JSZWYSCgoCaWQYASABKAkSDwoHdmVyc2lvbhgCIAEoCSKuAwoMTGF5ZXJSZWNlaXB0Eg0KBWxheWVyGAEgASgJEhAKCGluc3RhbmNlGAIgASgJEjEKBnN0YXR1cxgDIAEoDjIhLnR1cmJvLnJlc3VsdC52MS5Qcm9jZXNzaW5nU3RhdHVzEhIKCml0ZW1fY291bnQYBCABKA0SLAoIYXNzZXNzZWQYBSADKAsyGi50dXJiby5yZXN1bHQudjEuQnl0ZVJhbmdlEi8KCXJlcXVlc3RlZBgGIAMoCzIcLnR1cmJvLnJlc3VsdC52MS5EZXRlY3RvclJlZhIuCghleGVjdXRlZBgHIAMoCzIcLnR1cmJvLnJlc3VsdC52MS5EZXRlY3RvclJlZhI2Cgt0cnVuY2F0aW9ucxgIIAMoDjIhLnR1cmJvLnJlc3VsdC52MS5UcnVuY2F0aW9uUmVhc29uEi8KBWVycm9yGAkgASgLMhsudHVyYm8ucmVzdWx0LnYxLkxheWVyRXJyb3JIAIgBARIeChFwcm92ZW5hbmNlX2RpZ2VzdBgKIAEoDEgBiAEBQggKBl9lcnJvckIUChJfcHJvdmVuYW5jZV9kaWdlc3QijAMKEVByb2Nlc3NpbmdSZWNlaXB0Eh4KFnJlY2VpcHRfc2NoZW1hX3ZlcnNpb24YASABKA0SFQoNb3BlcmF0aW9uX2tleRgCIAEoCRISCgphdHRlbXB0X2lkGAMgASgJEi0KBWlucHV0GAQgASgLMh4udHVyYm8ucmVzdWx0LnYxLklucHV0SWRlbnRpdHkSNAoGcmVzdWx0GAUgASgLMh8udHVyYm8ucmVzdWx0LnYxLlJlc3VsdElkZW50aXR5SACIAQESMQoGc3RhdHVzGAYgASgOMiEudHVyYm8ucmVzdWx0LnYxLlByb2Nlc3NpbmdTdGF0dXMSEgoKaXRlbV9jb3VudBgHIAEoBBItCgZsYXllcnMYCCADKAsyHS50dXJiby5yZXN1bHQudjEuTGF5ZXJSZWNlaXB0EhcKD3JlcXVlc3RlZF9zdGVwcxgJIAMoCRIWCg5leGVjdXRlZF9zdGVwcxgKIAMoCRIVCg1jb25maWdfZGlnZXN0GAsgASgMQgkKB19yZXN1bHQiQwoNRXhlY3V0aW9uSW5mbxIPCgdiYWNrZW5kGAEgASgJEg4KBmRldmljZRgCIAEoCRIRCglwcmVjaXNpb24YAyABKAki3gEKD01vZGVsUHJvdmVuYW5jZRIRCgltb2RlbF9yZWYYASABKAkSEQoJYW5ub3RhdG9yGAIgASgJEg8KB3ZlcnNpb24YAyABKAkSEwoLY29uZmlnX2hhc2gYBCABKAQSLgoGYnVuZGxlGAUgASgLMhkudHVyYm8ucmVzdWx0LnYxLk1vZGVsUmVmSACIAQESNgoJZXhlY3V0aW9uGAYgASgLMh4udHVyYm8ucmVzdWx0LnYxLkV4ZWN1dGlvbkluZm9IAYgBAUIJCgdfYnVuZGxlQgwKCl9leGVjdXRpb24iTgoORnJhZ21lbnRNZW1iZXISDQoFaW5kZXgYASABKA0SLQoHb3V0Y29tZRgCIAEoDjIcLnR1cmJvLnJlc3VsdC52MS5TcGFuT3V0Y29tZSKLAQoMU3BhbkZyYWdtZW50Eg0KBXN0YXJ0GAEgASgNEgsKA2VuZBgCIAEoDRItCgdvdXRjb21lGAMgASgOMhwudHVyYm8ucmVzdWx0LnYxLlNwYW5PdXRjb21lEjAKB21lbWJlcnMYBCADKAsyHy50dXJiby5yZXN1bHQudjEuRnJhZ21lbnRNZW1iZXIiQQoPTGF5ZXJJbmRleEJvdW5kEg0KBWxheWVyGAEgASgJEhAKCGluc3RhbmNlGAIgASgJEg0KBWJvdW5kGAMgASgNIq8EChNUeXBlZERvY3VtZW50UmVzdWx0Eg4KBmRvY19pZBgBIAEoCRIQCghyYXdfdGV4dBgCIAEoCRIMCgR2aWV3GAMgASgNEiwKB29mZnNldHMYBCABKA4yGy50dXJiby5yZXN1bHQudjEuT2Zmc2V0TW9kZRIwCgZtb2RlbHMYBSADKAsyIC50dXJiby5yZXN1bHQudjEuTW9kZWxQcm92ZW5hbmNlEigKBXZpZXdzGAYgAygLMhkudHVyYm8ucmVzdWx0LnYxLlZpZXdJbmZvEisKBmxheWVycxgHIAMoCzIbLnR1cmJvLnJlc3VsdC52MS5UeXBlZExheWVyEi0KBWlucHV0GAggASgLMh4udHVyYm8ucmVzdWx0LnYxLklucHV0SWRlbnRpdHkSLwoGcmVzdWx0GAkgASgLMh8udHVyYm8ucmVzdWx0LnYxLlJlc3VsdElkZW50aXR5EjQKCnByb3ZlbmFuY2UYCiADKAsyIC50dXJiby5yZXN1bHQudjEuTGF5ZXJQcm92ZW5hbmNlEjAKB21hcHBpbmcYCyABKAsyHy50dXJiby5yZXN1bHQudjEuTWFwcGluZ1BheWxvYWQSMwoHcmVjZWlwdBgMIAEoCzIiLnR1cmJvLnJlc3VsdC52MS5Qcm9jZXNzaW5nUmVjZWlwdBIeChF3aXRoaGVsZF90ZXh0X2xlbhgUIAEoDUgAiAEBQhQKEl93aXRoaGVsZF90ZXh0X2xlbiLMAwoKVHlwZWRMYXllchIKCgJpZBgBIAEoCRIQCghpbnN0YW5jZRgCIAEoCRIwCgZzdGF0dXMYAyABKA4yIC50dXJiby5yZXN1bHQudjEuTGF5ZXJXaXJlU3RhdHVzEioKBXNjb3BlGAUgASgOMhsudHVyYm8ucmVzdWx0LnYxLkFuY2hvcktpbmQSEgoKbW9kZWxfcmVmcxgGIAMoCRIZCgxhbmNob3JfbGF5ZXIYByABKAlIAIgBARI1Cgthbm5vdGF0aW9ucxgIIAMoCzIgLnR1cmJvLnJlc3VsdC52MS5UeXBlZEFubm90YXRpb24SNwoNaW5kZXhlc19sYXllchgJIAMoCzIgLnR1cmJvLnJlc3VsdC52MS5MYXllckluZGV4Qm91bmQSGAoLbmF0aXZlX3ZpZXcYCiABKA1IAYgBARIvCgVlcnJvchgLIAEoCzIbLnR1cmJvLnJlc3VsdC52MS5MYXllckVycm9ySAKIAQESGAoQY3VzdG9tX3BpaV9raW5kcxgMIAMoCUIPCg1fYW5jaG9yX2xheWVyQg4KDF9uYXRpdmVfdmlld0IICgZfZXJyb3JKBAgEEAVSDXN0YXR1c19kZXRhaWwihQIKCkVudGl0eVZvdGUSEQoJYW5ub3RhdG9yGAEgASgJEhYKCW1vZGVsX3JlZhgCIAEoCUgAiAEBEiwKBXNwYW5zGAMgAygLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudBIYCgtwcm9iYWJpbGl0eRgEIAEoAUgBiAEBEiMKFmNhbGlicmF0ZWRfcHJvYmFiaWxpdHkYBSABKAFIAogBARIXCgphYnN0ZW50aW9uGAYgASgJSAOIAQFCDAoKX21vZGVsX3JlZkIOCgxfcHJvYmFiaWxpdHlCGQoXX2NhbGlicmF0ZWRfcHJvYmFiaWxpdHlCDQoLX2Fic3RlbnRpb24iMQoSU2VudGVuY2VBbm5vdGF0aW9uEg0KBWluZGV4GAEgASgNEgwKBHRleHQYAiABKAkiUgoPVG9rZW5Bbm5vdGF0aW9uEhsKDnNlbnRlbmNlX2luZGV4GAEgASgNSACIAQESDwoHc3VyZmFjZRgCIAEoCUIRCg9fc2VudGVuY2VfaW5kZXgihwEKDldvcmRBbm5vdGF0aW9uEhMKC3Rva2VuX3N0YXJ0GAEgASgNEhEKCXRva2VuX2VuZBgCIAEoDRIzCgdzdWJzcGFuGAMgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudEgAiAEBEgwKBGZvcm0YBCABKAlCCgoIX3N1YnNwYW4iYAoRU3Vid29yZEFubm90YXRpb24SEAoIdm9jYWJfaWQYASABKAUSGAoLdG9rZW5faW5kZXgYAiABKA1IAIgBARIPCgdzdXJmYWNlGAMgASgJQg4KDF90b2tlbl9pbmRleCJFChBQb3NUYWdBbm5vdGF0aW9uEg4KBnRhZ3NldBgBIAEoCRILCgN0YWcYAiABKAkSFAoMdGFyZ2V0X2luZGV4GAMgASgNIkEKF01vcnBoRmVhdHVyZXNBbm5vdGF0aW9uEhQKDHRhcmdldF9pbmRleBgBIAEoDRIQCghmZWF0dXJlcxgCIAEoCSI2Cg9MZW1tYUFubm90YXRpb24SFAoMdGFyZ2V0X2luZGV4GAEgASgNEg0KBWxlbW1hGAIgASgJIjMKDlN0ZW1Bbm5vdGF0aW9uEhMKC3Rva2VuX2luZGV4GAEgASgNEgwKBHN0ZW0YAiABKAkigAEKDlRlcm1Bbm5vdGF0aW9uEhMKC3Rva2VuX2luZGV4GAEgASgNEg8KB3N1cmZhY2UYAiABKAkSDgoGZm9sZGVkGAMgASgJEhEKBHN0ZW0YBCABKAlIAIgBARISCgVsZW1tYRgFIAEoCUgBiAEBQgcKBV9zdGVtQggKBl9sZW1tYSJGChJDaHVua1RhZ0Fubm90YXRpb24SDgoGdGFnc2V0GAEgASgJEgsKA3RhZxgCIAEoCRITCgt0b2tlbl9pbmRleBgDIAEoDSJdChVQaHJhc2VDaHVua0Fubm90YXRpb24SDgoGdGFnc2V0GAEgASgJEgwKBGtpbmQYAiABKAkSEwoLdG9rZW5fc3RhcnQYAyABKA0SEQoJdG9rZW5fZW5kGAQgASgNIpgBChBQaHJhc2VBbm5vdGF0aW9uEg4KBnRhZ3NldBgBIAEoCRIMCgRraW5kGAIgASgJEhMKC3Rva2VuX3N0YXJ0GAMgASgNEhEKCXRva2VuX2VuZBgEIAEoDRISCgpoZWFkX3Rva2VuGAUgASgNEhkKDHBhcmVudF9pbmRleBgGIAEoDUgAiAEBQg8KDV9wYXJlbnRfaW5kZXgidgoURGVwZW5kZW5jeUFubm90YXRpb24SDgoGdGFnc2V0GAEgASgJEhQKDHRhcmdldF9pbmRleBgCIAEoDRIXCgpoZWFkX2luZGV4GAMgASgNSACIAQESEAoIcmVsYXRpb24YBCABKAlCDQoLX2hlYWRfaW5kZXgitQEKEEVudGl0eUFubm90YXRpb24SDgoGdGFnc2V0GAEgASgJEgwKBGtpbmQYAiABKAkSEwoLdG9rZW5fc3RhcnQYAyABKA0SEQoJdG9rZW5fZW5kGAQgASgNEhEKCWNhbm9uaWNhbBgFIAEoCRISCgVrYl9pZBgGIAEoCUgAiAEBEioKBXZvdGVzGAcgAygLMhsudHVyYm8ucmVzdWx0LnYxLkVudGl0eVZvdGVCCAoGX2tiX2lkImEKElJlbGF0aW9uQW5ub3RhdGlvbhIOCgZ0YWdzZXQYASABKAkSDAoEa2luZBgCIAEoCRIWCg5zdWJqZWN0X2VudGl0eRgDIAEoDRIVCg1vYmplY3RfZW50aXR5GAQgASgNIvcBChZDb3JlZk1lbnRpb25Bbm5vdGF0aW9uEhYKDnNlbnRlbmNlX2luZGV4GAEgASgNEhMKC3Rva2VuX3N0YXJ0GAIgASgNEhEKCXRva2VuX2VuZBgDIAEoDRISCgpoZWFkX3Rva2VuGAQgASgNEhkKDGVudGl0eV9pbmRleBgFIAEoDUgAiAEBEgwKBGtpbmQYBiABKAkSDgoGb3JpZ2luGAcgASgJEg4KBm51bWJlchgIIAEoCRIOCgZnZW5kZXIYCSABKAkSDwoHYW5pbWFjeRgKIAEoCRIOCgZwZXJzb24YCyABKAlCDwoNX2VudGl0eV9pbmRleCJHChRDb3JlZkNoYWluQW5ub3RhdGlvbhIWCg5yZXByZXNlbnRhdGl2ZRgBIAEoCRIXCg9tZW50aW9uX2luZGljZXMYAiADKA0iJAoRU3BlYWtlckFubm90YXRpb24SDwoHc3BlYWtlchgBIAEoCSL3AQoWV29yZE5ldFNlbnNlQW5ub3RhdGlvbhITCgt0b2tlbl9pbmRleBgBIAEoDRIVCg1zeW5zZXRfb2Zmc2V0GAIgASgNEgsKA3BvcxgDIAEoCRIPCgdsZXhuYW1lGAQgASgJEg0KBWdsb3NzGAUgASgJEg4KBmxlbW1hcxgGIAMoCRIYChBoeXBlcm55bV9vZmZzZXRzGAcgAygNEhEKCXN5bnNldF9pZBgIIAEoCRIQCghzZW5zZV9pZBgJIAEoCRINCgVsZW1tYRgKIAEoCRISCgpzZW5zZV9yYW5rGAsgASgNEhIKCmxleGljb25faWQYDCABKAki+gEKGkxleGljYWxFeHBhbnNpb25Bbm5vdGF0aW9uEhMKC3Rva2VuX2luZGV4GAEgASgNEhAKCHN5bm9ueW1zGAIgAygJEhEKCWh5cGVybnltcxgDIAMoCRITCgtkZXJpdmF0aW9ucxgEIAMoCRITCgttb3JwaF9yb290cxgFIAMoCRIMCgR0ZXJtGAYgASgJEhAKCHJlbGF0aW9uGAcgASgNEg0KBWRlcHRoGAggASgNEhIKCnNlbnNlX3JhbmsYCSABKA0SDgoGd2VpZ2h0GAogASgBEhEKCXN5bnNldF9pZBgLIAEoCRISCgpsZXhpY29uX2lkGAwgASgJIoQBChZHbG9zc2FyeVRlcm1Bbm5vdGF0aW9uEhQKDGNhbm9uaWNhbF9pZBgBIAEoCRIXCg9wcmVmZXJyZWRfbGFiZWwYAiABKAkSEwoLZW50cnlfaW5kZXgYAyABKA0SEwoLdG9rZW5fc3RhcnQYBCABKA0SEQoJdG9rZW5fZW5kGAUgASgNIsgBChBOdW1iZXJBbm5vdGF0aW9uEjMKDG51bWVyYWxfc3BhbhgBIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5TcGFuRnJhZ21lbnQSEAoIbmVnYXRpdmUYAiABKAgSEwoLY29lZmZpY2llbnQYAyABKAkSDQoFc2NhbGUYBCABKA0SEgoKbXVsdGlwbGllchgFIAEoCRIOCgZtYXJrZXIYBiABKAkSEAoIbm90YXRpb24YByABKAkSEwoLcGxhaW5fdmFsdWUYCCABKAkiMQoRT3JkaW5hbEFubm90YXRpb24SDAoEYmFzZRgBIAEoCRIOCgZvZmZzZXQYAiABKAMibwoPTW9uZXlBbm5vdGF0aW9uEhAKCGN1cnJlbmN5GAEgASgJEhAKCG5lZ2F0aXZlGAIgASgIEhMKC2NvZWZmaWNpZW50GAMgASgJEg0KBXNjYWxlGAQgASgFEhQKDHBsYWluX2Ftb3VudBgFIAEoCSKBAQoSUXVhbnRpdHlBbm5vdGF0aW9uEgwKBHVuaXQYASABKAkSEAoIbmVnYXRpdmUYAiABKAgSEwoLY29lZmZpY2llbnQYAyABKAkSDQoFc2NhbGUYBCABKAUSEwoLcGxhaW5fdmFsdWUYBSABKAkSEgoKaXNfcGVyY2VudBgGIAEoCCJxChJUZW1wb3JhbEFubm90YXRpb24SEwoLZ3JhbnVsYXJpdHkYASABKAkSDgoGb3JpZ2luGAIgASgJEgsKA2lzbxgDIAEoCRIMCgR5ZWFyGAQgASgFEg4KBnBlcmlvZBgFIAEoDRILCgNkYXkYBiABKA0iwwEKDlRpbWVBbm5vdGF0aW9uEgwKBGtpbmQYASABKAkSDQoFZ3JhaW4YAiABKAkSEAoIbW9kaWZpZXIYAyABKAkSEgoFc3RhcnQYBCABKANIAIgBARIQCgNlbmQYBSABKANIAYgBARIYCgtkdXJhdGlvbl9tcxgGIAEoA0gCiAEBEhEKCWFtYmlndW91cxgHIAEoCBINCgV0aW1leBgIIAEoCUIICgZfc3RhcnRCBgoEX2VuZEIOCgxfZHVyYXRpb25fbXMiTwoWRG9jdW1lbnREYXRlQW5ub3RhdGlvbhIMCgR5ZWFyGAEgASgFEg0KBW1vbnRoGAIgASgNEgsKA2RheRgDIAEoDRILCgNpc28YBCABKAki7gEKD1BsYWNlUmVzb2x1dGlvbhIOCgZzb3VyY2UYASABKAkSEQoJcmVjb3JkX2lkGAIgASgJEhYKDmNhbm9uaWNhbF9uYW1lGAMgASgJEhUKDWZlYXR1cmVfY2xhc3MYBCABKAkSFAoMY291bnRyeV9jb2RlGAUgASgJEhMKC2FkbWluMV9jb2RlGAYgASgJEhMKC2FkbWluMl9jb2RlGAcgASgJEhAKCGxhdGl0dWRlGAggASgBEhEKCWxvbmdpdHVkZRgJIAEoARISCgpwb3B1bGF0aW9uGAogASgEEhAKCHRpbWV6b25lGAsgASgJIjwKElBsYWNlQW5jZXN0b3JQcm90bxIKCgJpZBgBIAEoCRIMCgRuYW1lGAIgASgJEgwKBGtpbmQYAyABKAkiSgoQQ29udGFpbm1lbnRDaGFpbhI2CglhbmNlc3RvcnMYASADKAsyIy50dXJiby5yZXN1bHQudjEuUGxhY2VBbmNlc3RvclByb3RvIjEKClJlZ2lvblZvdGUSFAoMY291bnRyeV9jb2RlGAEgASgJEg0KBXNoYXJlGAIgASgBIjEKEkxhbmd1YWdlQW5ub3RhdGlvbhIMCgRjb2RlGAEgASgJEg0KBXNjb3JlGAIgASgBIkoKFkxhbmd1YWdlU3BhbkFubm90YXRpb24SDAoEY29kZRgBIAEoCRIOCgZzY3JpcHQYAiABKAkSEgoKY29uZmlkZW5jZRgDIAEoASIyChJDYXRlZ29yeVNjb3JlUHJvdG8SDQoFbGFiZWwYASABKAkSDQoFc2NvcmUYAiABKAEikgEKEkNhdGVnb3J5QW5ub3RhdGlvbhIOCgZ0YWdzZXQYASABKAkSEgoKYmVzdF9sYWJlbBgCIAEoCRIRCglhYnN0YWluZWQYAyABKAgSEAoIc2VsZWN0ZWQYBCADKAkSMwoGc2NvcmVzGAUgAygLMiMudHVyYm8ucmVzdWx0LnYxLkNhdGVnb3J5U2NvcmVQcm90byJcChNTZW50aW1lbnRBbm5vdGF0aW9uEhAKCHBvbGFyaXR5GAEgASgJEhAKCHBvc2l0aXZlGAIgASgBEhAKCG5lZ2F0aXZlGAMgASgBEg8KB25ldXRyYWwYBCABKAEiTAoTTGFiZWxUb2tlblNwYW5Qcm90bxINCgVsYWJlbBgBIAEoCRITCgt0b2tlbl9zdGFydBgCIAEoDRIRCgl0b2tlbl9lbmQYAyABKA0ihgEKEUVtb3Rpb25Bbm5vdGF0aW9uEhMKC3RhcmdldF9raW5kGAEgASgJEhQKDHRhcmdldF9pbmRleBgCIAEoDRIOCgZzY29yZXMYAyADKAESNgoIZXZpZGVuY2UYBCADKAsyJC50dXJiby5yZXN1bHQudjEuTGFiZWxUb2tlblNwYW5Qcm90byKWAgoQQXNwZWN0QW5ub3RhdGlvbhIYCgt0b2tlbl9zdGFydBgBIAEoDUgAiAEBEhYKCXRva2VuX2VuZBgCIAEoDUgBiAEBEhsKDnNlbnRlbmNlX2luZGV4GAMgASgNSAKIAQESFQoIY2F0ZWdvcnkYBCABKAlIA4gBARIQCghwb2xhcml0eRgFIAEoCRIQCghwb3NpdGl2ZRgGIAEoARIQCghuZWdhdGl2ZRgHIAEoARIPCgduZXV0cmFsGAggASgBEhcKD29waW5pb25faW5kaWNlcxgJIAMoDUIOCgxfdG9rZW5fc3RhcnRCDAoKX3Rva2VuX2VuZEIRCg9fc2VudGVuY2VfaW5kZXhCCwoJX2NhdGVnb3J5InEKEU9waW5pb25Bbm5vdGF0aW9uEhMKC3Rva2VuX3N0YXJ0GAEgASgNEhEKCXRva2VuX2VuZBgCIAEoDRIQCghwb2xhcml0eRgDIAEoARIRCglpbnRlbnNpdHkYBCABKAESDwoHbmVnYXRlZBgFIAEoCCLzAQoUSW5jaXZpbGl0eUFubm90YXRpb24SEwoLdGFyZ2V0X2tpbmQYASABKAkSFAoMdGFyZ2V0X2luZGV4GAIgASgNEg4KBmluc3VsdBgDIAEoARIRCglwcm9mYW5pdHkYBCABKAESDgoGdGhyZWF0GAUgASgBEhcKD2lkZW50aXR5X2F0dGFjaxgGIAEoARITCgtpbnRvbGVyYW5jZRgHIAEoARIXCg9zZXZlcmVfdG94aWNpdHkYCCABKAESNgoIZXZpZGVuY2UYCSADKAsyJC50dXJiby5yZXN1bHQudjEuTGFiZWxUb2tlblNwYW5Qcm90byKlAgoUUHJhZ21hdGljc0Fubm90YXRpb24SFgoOc2VudGVuY2VfaW5kZXgYASABKA0SDQoFc2NvcmUYAiABKAESIQoUcG9zaXRpdmVfdG9rZW5fc3RhcnQYAyABKA1IAIgBARIfChJwb3NpdGl2ZV90b2tlbl9lbmQYBCABKA1IAYgBARIhChRuZWdhdGl2ZV90b2tlbl9zdGFydBgFIAEoDUgCiAEBEh8KEm5lZ2F0aXZlX3Rva2VuX2VuZBgGIAEoDUgDiAEBQhcKFV9wb3NpdGl2ZV90b2tlbl9zdGFydEIVChNfcG9zaXRpdmVfdG9rZW5fZW5kQhcKFV9uZWdhdGl2ZV90b2tlbl9zdGFydEIVChNfbmVnYXRpdmVfdG9rZW5fZW5kIlUKClBpaUtpbmRSZWYSLQoHYnVpbHRpbhgBIAEoDjIaLnR1cmJvLnJlc3VsdC52MS5QaWlLaW5kSWRIABIQCgZjdXN0b20YAiABKA1IAEIGCgRraW5kIi0KDlBpaURldGVjdG9yUmVmEgoKAmlkGAEgASgJEg8KB3ZlcnNpb24YAiABKAkiEgoQUGlpU2hhcGVFdmlkZW5jZSJfChNQaWlDaGVja3N1bUV2aWRlbmNlEjgKCWFsZ29yaXRobRgBIAEoDjIlLnR1cmJvLnJlc3VsdC52MS5QaWlDaGVja3N1bUFsZ29yaXRobRIOCgZwYXNzZWQYAiABKAgiTwoQUGlpTGFiZWxFdmlkZW5jZRIrCgRzcGFuGAEgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudBIOCgZwaHJhc2UYAiABKA0iMAoSUGlpRGVueUhpdEV2aWRlbmNlEgwKBGxpc3QYASABKA0SDAoEdGVybRgCIAEoDSIfCg9QaWlSdWxlRXZpZGVuY2USDAoEcnVsZRgBIAEoDSKiAgoLUGlpRXZpZGVuY2USMgoFc2hhcGUYASABKAsyIS50dXJiby5yZXN1bHQudjEuUGlpU2hhcGVFdmlkZW5jZUgAEjgKCGNoZWNrc3VtGAIgASgLMiQudHVyYm8ucmVzdWx0LnYxLlBpaUNoZWNrc3VtRXZpZGVuY2VIABIyCgVsYWJlbBgDIAEoCzIhLnR1cmJvLnJlc3VsdC52MS5QaWlMYWJlbEV2aWRlbmNlSAASNwoIZGVueV9oaXQYBCABKAsyIy50dXJiby5yZXN1bHQudjEuUGlpRGVueUhpdEV2aWRlbmNlSAASMAoEcnVsZRgFIAEoCzIgLnR1cmJvLnJlc3VsdC52MS5QaWlSdWxlRXZpZGVuY2VIAEIGCgRmYWN0IooBCghQaWlTY29yZRINCgV2YWx1ZRgBIAEoAhIrCgRraW5kGAIgASgOMh0udHVyYm8ucmVzdWx0LnYxLlBpaVNjb3JlS2luZBIuCgVtb2RlbBgDIAEoCzIfLnR1cmJvLnJlc3VsdC52MS5QaWlEZXRlY3RvclJlZhISCgpjYWxpYnJhdGVkGAQgASgIIoADCgpQaWlNZW50aW9uEhAKBGtpbmQYASABKAlCAhgBEhsKD2NvbmZpZGVuY2VfdGllchgCIAEoCUICGAESFgoKbm9ybWFsaXplZBgDIAEoCUICGAESKAoFdmFsdWUYBCABKAsyGS50dXJiby5yZXN1bHQudjEuUGlpVmFsdWUSMQoIZGV0ZWN0b3IYBSABKAsyHy50dXJiby5yZXN1bHQudjEuUGlpRGV0ZWN0b3JSZWYSLgoIZXZpZGVuY2UYBiADKAsyHC50dXJiby5yZXN1bHQudjEuUGlpRXZpZGVuY2USLQoFc2NvcmUYByABKAsyGS50dXJiby5yZXN1bHQudjEuUGlpU2NvcmVIAIgBARItCghraW5kX3JlZhgKIAEoCzIbLnR1cmJvLnJlc3VsdC52MS5QaWlLaW5kUmVmEjYKCmNvbmZpZGVuY2UYCyABKA4yIi50dXJiby5yZXN1bHQudjEuUGlpQ29uZmlkZW5jZVRpZXJCCAoGX3Njb3JlIsYBCghQaWlWYWx1ZRI1Cgh3aXRoaGVsZBgBIAEoCzIhLnR1cmJvLnJlc3VsdC52MS5QaWlWYWx1ZVdpdGhoZWxkSAASLwoFdG9rZW4YAiABKAsyHi50dXJiby5yZXN1bHQudjEuUGlpVmFsdWVUb2tlbkgAEjkKCmNpcGhlcnRleHQYAyABKAsyIy50dXJiby5yZXN1bHQudjEuUGlpVmFsdWVDaXBoZXJ0ZXh0SAASDwoFY2xlYXIYBCABKAlIAEIGCgRmb3JtIkYKEFBpaVZhbHVlV2l0aGhlbGQSMgoGcmVhc29uGAEgASgOMiIudHVyYm8ucmVzdWx0LnYxLlBpaVdpdGhoZWxkUmVhc29uIkMKDVBpaVZhbHVlVG9rZW4SDQoFdG9rZW4YASABKAwSDgoGa2V5X2lkGAIgASgJEhMKC2tleV92ZXJzaW9uGAMgASgNIl0KElBpaVZhbHVlQ2lwaGVydGV4dBISCgpjaXBoZXJ0ZXh0GAEgASgMEg4KBmtleV9pZBgCIAEoCRIOCgZzY2hlbWUYAyABKAkSEwoLa2V5X3ZlcnNpb24YBCABKA0ifAoQUGlpT3V0cHV0T3B0aW9ucxIWCg5yZXdyaXR0ZW5fdGV4dBgBIAEoCBIRCglmdWxsX3RleHQYAiABKAgSDgoGZGV0YWlsGAMgASgIEi0KBnZhbHVlcxgEIAEoDjIdLnR1cmJvLnJlc3VsdC52MS5QaWlWYWx1ZUZvcm0iPAoPUGlpQWxsb3dSdWxlUmVmEg4KBHRlcm0YASABKA1IABIRCgdwYXR0ZXJuGAIgASgNSABCBgoEcnVsZSJxChBQaWlBbGxvd2VkQnlSdWxlEi0KBXNjb3BlGAEgASgOMh4udHVyYm8ucmVzdWx0LnYxLlBpaUFsbG93U2NvcGUSLgoEcnVsZRgCIAEoCzIgLnR1cmJvLnJlc3VsdC52MS5QaWlBbGxvd1J1bGVSZWYi0wEKDlBpaURpc3Bvc2l0aW9uEg4KBGtlcHQYASABKA1IABI8Cg9hbGxvd2VkX2J5X3J1bGUYAiABKAsyIS50dXJiby5yZXN1bHQudjEuUGlpQWxsb3dlZEJ5UnVsZUgAEhMKCW5lc3RlZF9pbhgDIAEoDUgAEjwKEHJlamVjdGVkX2J5X3J1bGUYBCABKA4yIC50dXJiby5yZXN1bHQudjEuUGlpUmVqZWN0UmVhc29uSAASEQoHcmVwZWF0cxgFIAEoDUgAQg0KC2Rpc3Bvc2l0aW9uIsoCChZQaWlDYW5kaWRhdGVBbm5vdGF0aW9uEikKBGtpbmQYASABKAsyGy50dXJiby5yZXN1bHQudjEuUGlpS2luZFJlZhIxCghkZXRlY3RvchgCIAEoCzIfLnR1cmJvLnJlc3VsdC52MS5QaWlEZXRlY3RvclJlZhIuCghldmlkZW5jZRgDIAMoCzIcLnR1cmJvLnJlc3VsdC52MS5QaWlFdmlkZW5jZRItCgVzY29yZRgEIAEoCzIZLnR1cmJvLnJlc3VsdC52MS5QaWlTY29yZUgAiAEBEjQKC2Rpc3Bvc2l0aW9uGAUgASgLMh8udHVyYm8ucmVzdWx0LnYxLlBpaURpc3Bvc2l0aW9uEjMKB3BsYW5uZWQYBiABKAsyIi50dXJiby5yZXN1bHQudjEuUGlpUGxhbm5lZFJld3JpdGVCCAoGX3Njb3JlIrABChFQaWlQbGFubmVkUmV3cml0ZRIqCgRydWxlGAEgASgLMhwudHVyYm8ucmVzdWx0LnYxLlBpaUVkaXRSdWxlEg8KB3J1bGVfaWQYAiABKAkSLgoIc3RyYXRlZ3kYAyABKA4yHC50dXJiby5yZXN1bHQudjEuUGlpU3RyYXRlZ3kSFwoPcG9saWN5X3JldmlzaW9uGAQgASgEEhUKDXBvbGljeV9kaWdlc3QYBSABKAwiWgoLUGlpRWRpdFJ1bGUSFgoMZGVmYXVsdF9ydWxlGAEgASgISAASKwoEa2luZBgCIAEoCzIbLnR1cmJvLnJlc3VsdC52MS5QaWlLaW5kUmVmSABCBgoEcnVsZSLwAQoRUGlpRWRpdEFubm90YXRpb24SLQoGc291cmNlGAEgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudBIPCgdtZW1iZXJzGAIgAygNEioKBHJ1bGUYAyABKAsyHC50dXJiby5yZXN1bHQudjEuUGlpRWRpdFJ1bGUSLgoIc3RyYXRlZ3kYBCABKA4yHC50dXJiby5yZXN1bHQudjEuUGlpU3RyYXRlZ3kSFwoPcG9saWN5X3JldmlzaW9uGAUgASgEEg8KB3J1bGVfaWQYBiABKAkSFQoNcG9saWN5X2RpZ2VzdBgHIAEoDCLhAQoLUGlpUmlza0tpbmQSKQoEa2luZBgBIAEoCzIbLnR1cmJvLnJlc3VsdC52MS5QaWlLaW5kUmVmEhAKCG1lbnRpb25zGAIgASgNEi4KCHNldmVyaXR5GAMgASgOMhwudHVyYm8ucmVzdWx0LnYxLlBpaVNldmVyaXR5EhYKDndlaWdodGVkX2NvdW50GAQgASgCEhQKDGNvbnRyaWJ1dGlvbhgFIAEoAhI3CglzdWdnZXN0ZWQYBiABKA4yJC50dXJiby5yZXN1bHQudjEuUGlpU3VnZ2VzdGVkUmV3cml0ZSLdAQoRUGlpUmlza0Fubm90YXRpb24SDQoFc2NvcmUYASABKAISLAoFbGV2ZWwYAiABKA4yHS50dXJiby5yZXN1bHQudjEuUGlpUmlza0xldmVsEhAKCG1lbnRpb25zGAMgASgNEh4KFmRlbnNpdHlfcGVyXzEwMDBfY2hhcnMYBCABKAISKwoFa2luZHMYBSADKAsyHC50dXJiby5yZXN1bHQudjEuUGlpUmlza0tpbmQSHQoVc2V2ZXJpdHlfdGFibGVfZGlnZXN0GAYgASgGEg0KBWNoYXJzGAcgASgNIiwKCU5vaXNlU3BhbhIQCghzZXZlcml0eRgBIAEoCRINCgVzY29yZRgCIAEoASIcCgxUZXh0QXJ0aWZhY3QSDAoEa2luZBgBIAEoCSLmAQoNRW1iZWRkZWRBc3NldBIOCgZmb3JtYXQYASABKAkSEgoKbWVkaWFfdHlwZRgCIAEoCRIVCg1kZWNsYXJlZF90eXBlGAMgASgJEhAKCGVuY29kaW5nGAQgASgJEjMKDHBheWxvYWRfc3BhbhgFIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5TcGFuRnJhZ21lbnQSFQoNZGVjb2RlZF9ieXRlcxgGIAEoBBISCgV3aWR0aBgHIAEoDUgAiAEBEhMKBmhlaWdodBgIIAEoDUgBiAEBQggKBl93aWR0aEIJCgdfaGVpZ2h0Ij8KElNwZWxsaW5nQ29ycmVjdGlvbhISCgpzdWdnZXN0aW9uGAEgASgJEhUKDWVkaXRfZGlzdGFuY2UYAiABKA0iJQoVUGxhY2Vob2xkZXJBbm5vdGF0aW9uEgwKBGtpbmQYASABKAkiPQoPRW1vamlBbm5vdGF0aW9uEgwKBGtpbmQYASABKAkSDAoEbmFtZRgCIAEoCRIOCgZyZWdpb24YAyABKAkiKAoXUHJvdGVjdGVkU3BhbkFubm90YXRpb24SDQoFaW5kZXgYASABKA0iJgoWQWJicmV2aWF0aW9uQW5ub3RhdGlvbhIMCgRmb3JtGAEgASgJIlkKD01hdGNoQW5ub3RhdGlvbhITCgt0b2tlbl9zdGFydBgBIAEoDRIRCgl0b2tlbl9lbmQYAiABKA0SDQoFbGFiZWwYAyABKAkSDwoHcnVsZV9pZBgEIAEoDSJYChVCb2lsZXJwbGF0ZUFubm90YXRpb24SFAoMdGFyZ2V0X2luZGV4GAEgASgNEhEKBGtpbmQYAiABKAlIAIgBARINCgVzY29yZRgDIAEoAkIHCgVfa2luZCJAChRDb25maWRlbmNlQW5ub3RhdGlvbhIUCgx0YXJnZXRfaW5kZXgYASABKA0SEgoKY29uZmlkZW5jZRgCIAEoASJUChZTdG9wd29yZEZsYWdBbm5vdGF0aW9uEhMKC3Rva2VuX2luZGV4GAEgASgNEhAKCGxhbmd1YWdlGAIgASgJEhMKC2lzX3N0b3B3b3JkGAMgASgIIjcKEldvcmRUeXBlQW5ub3RhdGlvbhITCgt0b2tlbl9pbmRleBgBIAEoDRIMCgRraW5kGAIgASgJIpYBChFSZWFkaW5nQW5ub3RhdGlvbhITCgt0b2tlbl9pbmRleBgBIAEoDRIUCgdyZWFkaW5nGAIgASgJSACIAQESGgoNcHJvbnVuY2lhdGlvbhgDIAEoCUgBiAEBEhIKBWxlbW1hGAQgASgJSAKIAQFCCgoIX3JlYWRpbmdCEAoOX3Byb251bmNpYXRpb25CCAoGX2xlbW1hIlQKEUtleXdvcmRBbm5vdGF0aW9uEg4KBnBocmFzZRgBIAEoCRINCgVzY29yZRgCIAEoARIRCglhbGdvcml0aG0YAyABKAkSDQoFdGVybXMYBCABKA0iUgoYU291cmNlRW5jb2RpbmdBbm5vdGF0aW9uEhAKCGVuY29kaW5nGAEgASgJEhIKCmNvbmZpZGVuY2UYAiABKAISEAoIZXZpZGVuY2UYAyABKAkiVAoQVmVjdG9yQW5ub3RhdGlvbhIUCgx0YXJnZXRfaW5kZXgYASABKA0SCwoDZGltGAIgASgNEg0KBWVtcHR5GAMgASgIEg4KBnZhbHVlcxgEIAMoAiLOAgoPQ2h1bmtBbm5vdGF0aW9uEg0KBWxldmVsGAEgASgNEhkKDHBhcmVudF9pbmRleBgCIAEoDUgAiAEBEhYKDnNlbnRlbmNlX3N0YXJ0GAMgASgNEhQKDHNlbnRlbmNlX2VuZBgEIAEoDRISCgpjdXRfcmVhc29uGAUgASgJEgwKBGNvc3QYBiABKAMSFAoMbW9kZWxfdG9rZW5zGAcgASgNEhMKC292ZXJfYnVkZ2V0GAggASgIEhcKCnBhZ2Vfc3RhcnQYCSABKA1IAYgBARIVCghwYWdlX2VuZBgKIAEoDUgCiAEBEjkKCnBhZ2VfYm94ZXMYCyADKAsyJS50dXJiby5yZXN1bHQudjEuUGFnZUJvdW5kaW5nQm94UHJvdG9CDwoNX3BhcmVudF9pbmRleEINCgtfcGFnZV9zdGFydEILCglfcGFnZV9lbmQiXwoPTGV2ZWxBbm5vdGF0aW9uEgwKBGtpbmQYASABKAkSEgoKdW5pdF9pbmRleBgCIAEoDRIZCgxwYXJlbnRfaW5kZXgYAyABKA1IAIgBAUIPCg1fcGFyZW50X2luZGV4IkoKEkJvdW5kYXJ5QW5ub3RhdGlvbhIWCg5zZW50ZW5jZV9pbmRleBgBIAEoDRIMCgRjb3N0GAIgASgFEg4KBnJlYXNvbhgDIAEoCSJeChNTdHJ1Y3R1cmVBbm5vdGF0aW9uEgwKBGtpbmQYASABKAkSGQoMcGFyZW50X2luZGV4GAIgASgNSACIAQESDQoFZGVwdGgYAyABKA1CDwoNX3BhcmVudF9pbmRleCJyChBBbnN3ZXJBbm5vdGF0aW9uEg0KBXNjb3JlGAEgASgBEhMKC3Byb2JhYmlsaXR5GAIgASgBEhIKCm51bGxfc2NvcmUYAyABKAESGAoQbnVsbF9wcm9iYWJpbGl0eRgEIAEoARIMCgR0ZXh0GAUgASgJIlMKDkVkZ2VBbm5vdGF0aW9uEhIKCmZyb21faW5kZXgYASABKA0SEAoIdG9faW5kZXgYAiABKA0SDAoEa2luZBgDIAEoCRINCgVzY29yZRgEIAEoASIiCg9Hcm91cEFubm90YXRpb24SDwoHbWVtYmVycxgBIAMoDSLxAQoWTGlua2VkRW50aXR5QW5ub3RhdGlvbhIUCgxlbnRpdHlfaW5kZXgYASABKA0SEgoFa2JfaWQYAiABKAlIAIgBARITCgtwcm92aWRlcl9pZBgDIAEoCRIWCg5jYW5vbmljYWxfbmFtZRgEIAEoCRIVCg1tYXRjaGVkX2FsaWFzGAUgASgJEg0KBXNjb3JlGAYgASgBEhcKCm5pbF9yZWFzb24YByABKAlIAYgBARIYCgtuaWxfY2x1c3RlchgIIAEoDUgCiAEBQggKBl9rYl9pZEINCgtfbmlsX3JlYXNvbkIOCgxfbmlsX2NsdXN0ZXIijQEKFENhbGlicmF0ZWRBbm5vdGF0aW9uEhQKDHRhcmdldF9pbmRleBgBIAEoDRILCgNyYXcYAiADKAESEgoKY2FsaWJyYXRlZBgDIAMoARIWCg5wcmVkaWN0aW9uX3NldBgEIAMoDRIXCgphYnN0ZW50aW9uGAUgASgJSACIAQFCDQoLX2Fic3RlbnRpb24iMwobVG9rZW5WZWN0b3JXaW5kb3dBbm5vdGF0aW9uEhQKDHdpbmRvd19pbmRleBgBIAEoDSJYChNVdGYxNlNwYW5Bbm5vdGF0aW9uEhQKDHRhcmdldF9pbmRleBgBIAEoDRIPCgdzdGFydDE2GAIgASgNEg0KBWVuZDE2GAMgASgNEgsKA2N1dBgEIAEoCCKXAQoSTW9kYWxpdHlBbm5vdGF0aW9uEhAKCHBvbGFyaXR5GAEgASgJEhEKCWVwaXN0ZW1pYxgCIAEoCRIPCgdkZW9udGljGAMgASgJEi8KA2N1ZRgEIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5TcGFuRnJhZ21lbnRIAIgBARISCgpjb25maWRlbmNlGAUgASgCQgYKBF9jdWUiqgEKGENvbnNvbGlkYXRlZE1lbnRpb25Qcm90bxIrCgRzcGFuGAEgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudBIZCgxlbnRpdHlfaW5kZXgYAiABKA1IAIgBARIZChFpc19yZXByZXNlbnRhdGl2ZRgDIAEoCBIRCgRyb2xlGAQgASgJSAGIAQFCDwoNX2VudGl0eV9pbmRleEIHCgVfcm9sZSJkChNFbnRpdHlSZWxhdGlvblByb3RvEg8KB3N1YmplY3QYASABKA0SDgoGb2JqZWN0GAIgASgNEgwKBGtpbmQYAyABKAkSDQoFc2NvcmUYBCABKAISDwoHc3VwcG9ydBgFIAEoDSLsAwoYRG9jdW1lbnRFbnRpdHlBbm5vdGF0aW9uEhYKDmNhbm9uaWNhbF9uYW1lGAEgASgJEgwKBGtpbmQYAiABKAkSEgoFa2JfaWQYAyABKAlIAIgBARIVCghsYXRpdHVkZRgEIAEoAkgBiAEBEhYKCWxvbmdpdHVkZRgFIAEoAkgCiAEBEhkKDGNvdW50cnlfY29kZRgGIAEoCUgDiAEBEjsKCG1lbnRpb25zGAcgAygLMikudHVyYm8ucmVzdWx0LnYxLkNvbnNvbGlkYXRlZE1lbnRpb25Qcm90bxIVCg1zdWJqZWN0X2NvdW50GAggASgNEhQKDG9iamVjdF9jb3VudBgJIAEoDRIWCg5tb2RpZmllcl9jb3VudBgKIAEoDRITCgtvdGhlcl9jb3VudBgLIAEoDRIWCg5tZWFuX3NlbnRpbWVudBgMIAEoAhIaChJzZW50aW1lbnRfbWVudGlvbnMYDSABKA0SEgoKY29uZmlkZW5jZRgOIAEoAhI3CglyZWxhdGlvbnMYDyADKAsyJC50dXJiby5yZXN1bHQudjEuRW50aXR5UmVsYXRpb25Qcm90b0IICgZfa2JfaWRCCwoJX2xhdGl0dWRlQgwKCl9sb25naXR1ZGVCDwoNX2NvdW50cnlfY29kZSKXAQoSRXZlbnRBcmd1bWVudFByb3RvEisKBHNwYW4YASABKAsyHS50dXJiby5yZXN1bHQudjEuU3BhbkZyYWdtZW50EgwKBHJvbGUYAiABKAkSFgoOYW5jaG9yZWRfbGF5ZXIYAyABKAkSGwoOYW5jaG9yZWRfaW5kZXgYBCABKA1IAIgBAUIRCg9fYW5jaG9yZWRfaW5kZXgiowEKD0V2ZW50QW5ub3RhdGlvbhIaCg10cmlnZ2VyX2xlbW1hGAEgASgJSACIAQESFgoOc2VudGVuY2VfaW5kZXgYAiABKA0SEgoKY29uZmlkZW5jZRgDIAEoAhI2Cglhcmd1bWVudHMYBCADKAsyIy50dXJiby5yZXN1bHQudjEuRXZlbnRBcmd1bWVudFByb3RvQhAKDl90cmlnZ2VyX2xlbW1hIkQKDUVkdUFubm90YXRpb24SDQoFaW5kZXgYASABKA0SFgoOc2VudGVuY2VfaW5kZXgYAiABKA0SDAoEdGV4dBgDIAEoCSKmAQoUU3BhcnNlVGVybUVudHJ5UHJvdG8SDwoHdGVybV9pZBgBIAEoDRIOCgZ3ZWlnaHQYAiABKAISMAoEc3BhbhgDIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5TcGFuRnJhZ21lbnRIAIgBARIyCgtvY2N1cnJlbmNlcxgEIAMoCzIdLnR1cmJvLnJlc3VsdC52MS5TcGFuRnJhZ21lbnRCBwoFX3NwYW4iZQoVU3BhcnNlVGVybXNBbm5vdGF0aW9uEhQKDHRhcmdldF9pbmRleBgBIAEoDRI2CgdlbnRyaWVzGAIgAygLMiUudHVyYm8ucmVzdWx0LnYxLlNwYXJzZVRlcm1FbnRyeVByb3RvIo8BChVBbHRlcm5hdGl2ZUFubm90YXRpb24SEAoIcG9zaXRpb24YASABKA0SFwoPcG9zaXRpb25fbGVuZ3RoGAIgASgNEhEKCW1vZGVfYml0cxgDIAEoDRISCgVzY29yZRgEIAEoAUgAiAEBEhEKBGZvcm0YBSABKAlIAYgBAUIICgZfc2NvcmVCBwoFX2Zvcm0icgoTTkJlc3RDYW5kaWRhdGVQcm90bxIMCgRyYW5rGAEgASgNEhYKDnNlcXVlbmNlX3Njb3JlGAIgASgBEgsKA3RhZxgDIAEoCRIYCgtwcm9iYWJpbGl0eRgEIAEoAUgAiAEBQg4KDF9wcm9iYWJpbGl0eSJxCg9OQmVzdEFubm90YXRpb24SFAoMdGFyZ2V0X2luZGV4GAEgASgNEg4KBnRhZ3NldBgCIAEoCRI4CgpjYW5kaWRhdGVzGAMgAygLMiQudHVyYm8ucmVzdWx0LnYxLk5CZXN0Q2FuZGlkYXRlUHJvdG8ikwEKE1RhYmxlQ2VsbEFubm90YXRpb24SEAoIdGFibGVfaWQYASABKA0SEQoJcm93X2luZGV4GAIgASgNEhEKCWNvbF9pbmRleBgDIAEoDRIQCghyb3dfc3BhbhgEIAEoDRIQCghjb2xfc3BhbhgFIAEoDRIRCglpc19oZWFkZXIYBiABKAgSDQoFYWxpZ24YByABKAki2QEKFktleVZhbHVlUGFpckFubm90YXRpb24SLwoIa2V5X3NwYW4YASABKAsyHS50dXJiby5yZXN1bHQudjEuU3BhbkZyYWdtZW50EjUKDnNlcGFyYXRvcl9zcGFuGAIgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudBIxCgp2YWx1ZV9zcGFuGAMgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudBIQCghrZXlfdGV4dBgEIAEoCRISCgp2YWx1ZV90ZXh0GAUgASgJIrEBChhFbnRpdHlTYWxpZW5jZUFubm90YXRpb24SFAoMZW50aXR5X2luZGV4GAEgASgNEhAKCHNhbGllbmNlGAIgASgCEgwKBHJhbmsYAyABKA0SFgoOcG9zaXRpb25fc2NvcmUYBCABKAISFAoMc3ludGF4X3Njb3JlGAUgASgCEhcKD2ZyZXF1ZW5jeV9zY29yZRgGIAEoAhIYChBjZW50cmFsaXR5X3Njb3JlGAcgASgCIqgBChpDaHVua1RhYmxlSGVhZGVyQW5ub3RhdGlvbhITCgtjaHVua19pbmRleBgBIAEoDRIQCgh0YWJsZV9pZBgCIAEoDRIXCg9pc19jb250aW51YXRpb24YAyABKAgSMwoMaGVhZGVyX2NlbGxzGAQgAygLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudBIVCg1oZWFkZXJfcHJlZml4GAUgASgJIoABChVSZWNvcmRWYWx1ZUFubm90YXRpb24SFAoMcmVjb3JkX2luZGV4GAEgASgNEhMKC3JlY29yZF90eXBlGAIgASgJEhAKCGluc3RhbmNlGAMgASgNEgwKBHBhdGgYBCABKAkSDQoFc2NvcmUYBSABKAESDQoFbG9naXQYBiABKAEi/QQKGFF1YWxpdHlNZXRyaWNzQW5ub3RhdGlvbhIcCg9jb21wb3NpdGVfc2NvcmUYASABKAJIAIgBARIjChZtdGxkX2xleGljYWxfZGl2ZXJzaXR5GAIgASgCSAGIAQESIQoUZmxlc2NoX2tpbmNhaWRfZ3JhZGUYAyABKAJIAogBARIoChthdXRvbWF0ZWRfcmVhZGFiaWxpdHlfaW5kZXgYBCABKAJIA4gBARIfChJjb2xlbWFuX2xpYXVfaW5kZXgYBSABKAJIBIgBARIgChNpbmZvcm1hdGlvbl9kZW5zaXR5GAYgASgCSAWIAQESGAoLYWxwaGFfcmF0aW8YByABKAJIBogBARIYCgtkaWdpdF9yYXRpbxgIIAEoAkgHiAEBEhkKDHN5bWJvbF9yYXRpbxgJIAEoAkgIiAEBEh0KEHdoaXRlc3BhY2VfcmF0aW8YCiABKAJICYgBARIeChFyZXBsYWNlbWVudF9yYXRpbxgLIAEoAkgKiAEBEg0KBWZsYWdzGAwgASgNQhIKEF9jb21wb3NpdGVfc2NvcmVCGQoXX210bGRfbGV4aWNhbF9kaXZlcnNpdHlCFwoVX2ZsZXNjaF9raW5jYWlkX2dyYWRlQh4KHF9hdXRvbWF0ZWRfcmVhZGFiaWxpdHlfaW5kZXhCFQoTX2NvbGVtYW5fbGlhdV9pbmRleEIWChRfaW5mb3JtYXRpb25fZGVuc2l0eUIOCgxfYWxwaGFfcmF0aW9CDgoMX2RpZ2l0X3JhdGlvQg8KDV9zeW1ib2xfcmF0aW9CEwoRX3doaXRlc3BhY2VfcmF0aW9CFAoSX3JlcGxhY2VtZW50X3JhdGlvIv0CChtHcmFtbWFyRGlhZ25vc3RpY0Fubm90YXRpb24SDwoHcnVsZV9pZBgBIAEoCRJMCghjYXRlZ29yeRgCIAEoDjI1LnR1cmJvLnJlc3VsdC52MS5HcmFtbWFyRGlhZ25vc3RpY0Fubm90YXRpb24uQ2F0ZWdvcnlIAIgBARIPCgdtZXNzYWdlGAMgASgJEhQKDHJlcGxhY2VtZW50cxgEIAMoCRISCgpjb25maWRlbmNlGAUgASgCIrYBCghDYXRlZ29yeRIaChZTVUJKRUNUX1ZFUkJfQUdSRUVNRU5UEAASHQoZREVURVJNSU5FUl9OT1VOX0FHUkVFTUVOVBABEhUKEUFSVElDTEVfUEhPTkVUSUNTEAISFwoTQVVYSUxJQVJZX1ZFUkJfRk9STRADEhIKDlJFUEVBVEVEX1RPS0VOEAQSEwoPQ0FTRV9HT1ZFUk5NRU5UEAUSFgoSQ0xBVVNFX1BVTkNUVUFUSU9OEAZCCwoJX2NhdGVnb3J5Ik4KD0V2aWRlbmNlU3VwcG9ydBINCgVsYXllchgBIAEoCRIQCghpbnN0YW5jZRgCIAEoCRINCgVzdGFydBgDIAEoDRILCgNlbmQYBCABKA0i6wEKEkV2aWRlbmNlQW5ub3RhdGlvbhIUCgx0YXJnZXRfaW5kZXgYASABKA0SFgoJcmF3X3Njb3JlGAIgASgBSACIAQESIwoWY2FsaWJyYXRlZF9wcm9iYWJpbGl0eRgDIAEoAUgBiAEBEhcKCmFic3RlbnRpb24YBCABKAlIAogBARIxCgdzdXBwb3J0GAUgAygLMiAudHVyYm8ucmVzdWx0LnYxLkV2aWRlbmNlU3VwcG9ydEIMCgpfcmF3X3Njb3JlQhkKF19jYWxpYnJhdGVkX3Byb2JhYmlsaXR5Qg0KC19hYnN0ZW50aW9uIsACChNRdW90YXRpb25Bbm5vdGF0aW9uEjwKBGtpbmQYASABKA4yKS50dXJiby5yZXN1bHQudjEuUXVvdGF0aW9uQW5ub3RhdGlvbi5LaW5kSACIAQESFwoKY29uZmlkZW5jZRgCIAEoAkgBiAEBEi4KB3NwZWFrZXIYAyABKAsyHS50dXJiby5yZXN1bHQudjEuU3BhbkZyYWdtZW50Ei8KCGN1ZV92ZXJiGAQgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNwYW5GcmFnbWVudBIaCg1zcGVha2VyX2NoYWluGAUgASgNSAKIAQEiKwoES2luZBIKCgZESVJFQ1QQABIMCghJTkRJUkVDVBABEgkKBU1JWEVEEAJCBwoFX2tpbmRCDQoLX2NvbmZpZGVuY2VCEAoOX3NwZWFrZXJfY2hhaW4iVAoUUGFnZUJvdW5kaW5nQm94UHJvdG8SDAoEcGFnZRgBIAEoDRIKCgJ4MBgCIAEoAhIKCgJ5MBgDIAEoAhIKCgJ4MRgEIAEoAhIKCgJ5MRgFIAEoAiJTChNMYXlvdXRCb3hBbm5vdGF0aW9uEgwKBHBhZ2UYASABKA0SCgoCeDAYAiABKAISCgoCeTAYAyABKAISCgoCeDEYBCABKAISCgoCeTEYBSABKAIitQEKF0NodW5rQ2l0YXRpb25Bbm5vdGF0aW9uEhMKC2NodW5rX2luZGV4GAEgASgNEhcKCnBhZ2Vfc3RhcnQYAiABKA1IAIgBARIVCghwYWdlX2VuZBgDIAEoDUgBiAEBEjkKCnBhZ2VfYm94ZXMYBCADKAsyJS50dXJiby5yZXN1bHQudjEuUGFnZUJvdW5kaW5nQm94UHJvdG9CDQoLX3BhZ2Vfc3RhcnRCCwoJX3BhZ2VfZW5kIu4qCg9UeXBlZEFubm90YXRpb24SLAoFc3BhbnMYASADKAsyHS50dXJiby5yZXN1bHQudjEuU3BhbkZyYWdtZW50EhgKC3Byb2JhYmlsaXR5GAIgASgBSAGIAQESFgoJbW9kZWxfcmVmGAMgASgJSAKIAQESNwoIc2VudGVuY2UYCiABKAsyIy50dXJiby5yZXN1bHQudjEuU2VudGVuY2VBbm5vdGF0aW9uSAASMQoFdG9rZW4YCyABKAsyIC50dXJiby5yZXN1bHQudjEuVG9rZW5Bbm5vdGF0aW9uSAASLwoEd29yZBgMIAEoCzIfLnR1cmJvLnJlc3VsdC52MS5Xb3JkQW5ub3RhdGlvbkgAEjUKB3N1YndvcmQYDSABKAsyIi50dXJiby5yZXN1bHQudjEuU3Vid29yZEFubm90YXRpb25IABI0Cgdwb3NfdGFnGA4gASgLMiEudHVyYm8ucmVzdWx0LnYxLlBvc1RhZ0Fubm90YXRpb25IABJCCg5tb3JwaF9mZWF0dXJlcxgPIAEoCzIoLnR1cmJvLnJlc3VsdC52MS5Nb3JwaEZlYXR1cmVzQW5ub3RhdGlvbkgAEjEKBWxlbW1hGBAgASgLMiAudHVyYm8ucmVzdWx0LnYxLkxlbW1hQW5ub3RhdGlvbkgAEi8KBHN0ZW0YESABKAsyHy50dXJiby5yZXN1bHQudjEuU3RlbUFubm90YXRpb25IABIvCgR0ZXJtGBIgASgLMh8udHVyYm8ucmVzdWx0LnYxLlRlcm1Bbm5vdGF0aW9uSAASOAoJY2h1bmtfdGFnGBMgASgLMiMudHVyYm8ucmVzdWx0LnYxLkNodW5rVGFnQW5ub3RhdGlvbkgAEj4KDHBocmFzZV9jaHVuaxgUIAEoCzImLnR1cmJvLnJlc3VsdC52MS5QaHJhc2VDaHVua0Fubm90YXRpb25IABIzCgZwaHJhc2UYFSABKAsyIS50dXJiby5yZXN1bHQudjEuUGhyYXNlQW5ub3RhdGlvbkgAEjsKCmRlcGVuZGVuY3kYFiABKAsyJS50dXJiby5yZXN1bHQudjEuRGVwZW5kZW5jeUFubm90YXRpb25IABIzCgZlbnRpdHkYFyABKAsyIS50dXJiby5yZXN1bHQudjEuRW50aXR5QW5ub3RhdGlvbkgAEjcKCHJlbGF0aW9uGBggASgLMiMudHVyYm8ucmVzdWx0LnYxLlJlbGF0aW9uQW5ub3RhdGlvbkgAEkAKDWNvcmVmX21lbnRpb24YGSABKAsyJy50dXJiby5yZXN1bHQudjEuQ29yZWZNZW50aW9uQW5ub3RhdGlvbkgAEjwKC2NvcmVmX2NoYWluGBogASgLMiUudHVyYm8ucmVzdWx0LnYxLkNvcmVmQ2hhaW5Bbm5vdGF0aW9uSAASNQoHc3BlYWtlchgbIAEoCzIiLnR1cmJvLnJlc3VsdC52MS5TcGVha2VyQW5ub3RhdGlvbkgAEkAKDXdvcmRuZXRfc2Vuc2UYHCABKAsyJy50dXJiby5yZXN1bHQudjEuV29yZE5ldFNlbnNlQW5ub3RhdGlvbkgAEkgKEWxleGljYWxfZXhwYW5zaW9uGB0gASgLMisudHVyYm8ucmVzdWx0LnYxLkxleGljYWxFeHBhbnNpb25Bbm5vdGF0aW9uSAASQAoNZ2xvc3NhcnlfdGVybRgeIAEoCzInLnR1cmJvLnJlc3VsdC52MS5HbG9zc2FyeVRlcm1Bbm5vdGF0aW9uSAASMwoGbnVtYmVyGB8gASgLMiEudHVyYm8ucmVzdWx0LnYxLk51bWJlckFubm90YXRpb25IABI1CgdvcmRpbmFsGCAgASgLMiIudHVyYm8ucmVzdWx0LnYxLk9yZGluYWxBbm5vdGF0aW9uSAASMQoFbW9uZXkYISABKAsyIC50dXJiby5yZXN1bHQudjEuTW9uZXlBbm5vdGF0aW9uSAASNwoIcXVhbnRpdHkYIiABKAsyIy50dXJiby5yZXN1bHQudjEuUXVhbnRpdHlBbm5vdGF0aW9uSAASNwoIdGVtcG9yYWwYIyABKAsyIy50dXJiby5yZXN1bHQudjEuVGVtcG9yYWxBbm5vdGF0aW9uSAASLwoEdGltZRgkIAEoCzIfLnR1cmJvLnJlc3VsdC52MS5UaW1lQW5ub3RhdGlvbkgAEkAKDWRvY3VtZW50X2RhdGUYJSABKAsyJy50dXJiby5yZXN1bHQudjEuRG9jdW1lbnREYXRlQW5ub3RhdGlvbkgAEjEKBXBsYWNlGCYgASgLMiAudHVyYm8ucmVzdWx0LnYxLlBsYWNlUmVzb2x1dGlvbkgAEjgKC2NvbnRhaW5tZW50GCcgASgLMiEudHVyYm8ucmVzdWx0LnYxLkNvbnRhaW5tZW50Q2hhaW5IABIyCgtyZWdpb25fdm90ZRgoIAEoCzIbLnR1cmJvLnJlc3VsdC52MS5SZWdpb25Wb3RlSAASNwoIbGFuZ3VhZ2UYKSABKAsyIy50dXJiby5yZXN1bHQudjEuTGFuZ3VhZ2VBbm5vdGF0aW9uSAASQAoNbGFuZ3VhZ2Vfc3BhbhgqIAEoCzInLnR1cmJvLnJlc3VsdC52MS5MYW5ndWFnZVNwYW5Bbm5vdGF0aW9uSAASNwoIY2F0ZWdvcnkYKyABKAsyIy50dXJiby5yZXN1bHQudjEuQ2F0ZWdvcnlBbm5vdGF0aW9uSAASOQoJc2VudGltZW50GCwgASgLMiQudHVyYm8ucmVzdWx0LnYxLlNlbnRpbWVudEFubm90YXRpb25IABI1CgdlbW90aW9uGC0gASgLMiIudHVyYm8ucmVzdWx0LnYxLkVtb3Rpb25Bbm5vdGF0aW9uSAASMwoGYXNwZWN0GC4gASgLMiEudHVyYm8ucmVzdWx0LnYxLkFzcGVjdEFubm90YXRpb25IABI1CgdvcGluaW9uGC8gASgLMiIudHVyYm8ucmVzdWx0LnYxLk9waW5pb25Bbm5vdGF0aW9uSAASOwoKaW5jaXZpbGl0eRgwIAEoCzIlLnR1cmJvLnJlc3VsdC52MS5JbmNpdmlsaXR5QW5ub3RhdGlvbkgAEjsKCnByYWdtYXRpY3MYMSABKAsyJS50dXJiby5yZXN1bHQudjEuUHJhZ21hdGljc0Fubm90YXRpb25IABIqCgNwaWkYMiABKAsyGy50dXJiby5yZXN1bHQudjEuUGlpTWVudGlvbkgAEisKBW5vaXNlGDMgASgLMhoudHVyYm8ucmVzdWx0LnYxLk5vaXNlU3BhbkgAEjEKCGFydGlmYWN0GDQgASgLMh0udHVyYm8ucmVzdWx0LnYxLlRleHRBcnRpZmFjdEgAEi8KBWFzc2V0GDUgASgLMh4udHVyYm8ucmVzdWx0LnYxLkVtYmVkZGVkQXNzZXRIABI5Cgpjb3JyZWN0aW9uGDYgASgLMiMudHVyYm8ucmVzdWx0LnYxLlNwZWxsaW5nQ29ycmVjdGlvbkgAEj0KC3BsYWNlaG9sZGVyGDcgASgLMiYudHVyYm8ucmVzdWx0LnYxLlBsYWNlaG9sZGVyQW5ub3RhdGlvbkgAEjEKBWVtb2ppGDggASgLMiAudHVyYm8ucmVzdWx0LnYxLkVtb2ppQW5ub3RhdGlvbkgAEkIKDnByb3RlY3RlZF9zcGFuGDkgASgLMigudHVyYm8ucmVzdWx0LnYxLlByb3RlY3RlZFNwYW5Bbm5vdGF0aW9uSAASPwoMYWJicmV2aWF0aW9uGDogASgLMicudHVyYm8ucmVzdWx0LnYxLkFiYnJldmlhdGlvbkFubm90YXRpb25IABI2CgpydWxlX21hdGNoGDsgASgLMiAudHVyYm8ucmVzdWx0LnYxLk1hdGNoQW5ub3RhdGlvbkgAEj0KC2JvaWxlcnBsYXRlGDwgASgLMiYudHVyYm8ucmVzdWx0LnYxLkJvaWxlcnBsYXRlQW5ub3RhdGlvbkgAEjsKCmNvbmZpZGVuY2UYPSABKAsyJS50dXJiby5yZXN1bHQudjEuQ29uZmlkZW5jZUFubm90YXRpb25IABJACg1zdG9wd29yZF9mbGFnGD4gASgLMicudHVyYm8ucmVzdWx0LnYxLlN0b3B3b3JkRmxhZ0Fubm90YXRpb25IABI4Cgl3b3JkX3R5cGUYPyABKAsyIy50dXJiby5yZXN1bHQudjEuV29yZFR5cGVBbm5vdGF0aW9uSAASNQoHcmVhZGluZxhAIAEoCzIiLnR1cmJvLnJlc3VsdC52MS5SZWFkaW5nQW5ub3RhdGlvbkgAEjUKB2tleXdvcmQYQSABKAsyIi50dXJiby5yZXN1bHQudjEuS2V5d29yZEFubm90YXRpb25IABJECg9zb3VyY2VfZW5jb2RpbmcYQiABKAsyKS50dXJiby5yZXN1bHQudjEuU291cmNlRW5jb2RpbmdBbm5vdGF0aW9uSAASMwoGdmVjdG9yGEMgASgLMiEudHVyYm8ucmVzdWx0LnYxLlZlY3RvckFubm90YXRpb25IABIxCgVjaHVuaxhEIAEoCzIgLnR1cmJvLnJlc3VsdC52MS5DaHVua0Fubm90YXRpb25IABI2CgpsZXZlbF91bml0GEUgASgLMiAudHVyYm8ucmVzdWx0LnYxLkxldmVsQW5ub3RhdGlvbkgAEjcKCGJvdW5kYXJ5GEYgASgLMiMudHVyYm8ucmVzdWx0LnYxLkJvdW5kYXJ5QW5ub3RhdGlvbkgAEjkKCXN0cnVjdHVyZRhHIAEoCzIkLnR1cmJvLnJlc3VsdC52MS5TdHJ1Y3R1cmVBbm5vdGF0aW9uSAASMwoGYW5zd2VyGEggASgLMiEudHVyYm8ucmVzdWx0LnYxLkFuc3dlckFubm90YXRpb25IABIvCgRlZGdlGEkgASgLMh8udHVyYm8ucmVzdWx0LnYxLkVkZ2VBbm5vdGF0aW9uSAASMQoFZ3JvdXAYSiABKAsyIC50dXJiby5yZXN1bHQudjEuR3JvdXBBbm5vdGF0aW9uSAASQAoNbGlua2VkX2VudGl0eRhLIAEoCzInLnR1cmJvLnJlc3VsdC52MS5MaW5rZWRFbnRpdHlBbm5vdGF0aW9uSAASOwoKY2FsaWJyYXRlZBhMIAEoCzIlLnR1cmJvLnJlc3VsdC52MS5DYWxpYnJhdGVkQW5ub3RhdGlvbkgAEksKE3Rva2VuX3ZlY3Rvcl93aW5kb3cYTSABKAsyLC50dXJiby5yZXN1bHQudjEuVG9rZW5WZWN0b3JXaW5kb3dBbm5vdGF0aW9uSAASOgoKdXRmMTZfc3BhbhhOIAEoCzIkLnR1cmJvLnJlc3VsdC52MS5VdGYxNlNwYW5Bbm5vdGF0aW9uSAASNwoIbW9kYWxpdHkYTyABKAsyIy50dXJiby5yZXN1bHQudjEuTW9kYWxpdHlBbm5vdGF0aW9uSAASRAoPZG9jdW1lbnRfZW50aXR5GFAgASgLMikudHVyYm8ucmVzdWx0LnYxLkRvY3VtZW50RW50aXR5QW5ub3RhdGlvbkgAEjEKBWV2ZW50GFEgASgLMiAudHVyYm8ucmVzdWx0LnYxLkV2ZW50QW5ub3RhdGlvbkgAEi0KA2VkdRhSIAEoCzIeLnR1cmJvLnJlc3VsdC52MS5FZHVBbm5vdGF0aW9uSAASPgoMc3BhcnNlX3Rlcm1zGFMgASgLMiYudHVyYm8ucmVzdWx0LnYxLlNwYXJzZVRlcm1zQW5ub3RhdGlvbkgAEj0KC2FsdGVybmF0aXZlGFQgASgLMiYudHVyYm8ucmVzdWx0LnYxLkFsdGVybmF0aXZlQW5ub3RhdGlvbkgAEjEKBW5iZXN0GFUgASgLMiAudHVyYm8ucmVzdWx0LnYxLk5CZXN0QW5ub3RhdGlvbkgAEjoKCnRhYmxlX2NlbGwYViABKAsyJC50dXJiby5yZXN1bHQudjEuVGFibGVDZWxsQW5ub3RhdGlvbkgAEkEKDmtleV92YWx1ZV9wYWlyGFcgASgLMicudHVyYm8ucmVzdWx0LnYxLktleVZhbHVlUGFpckFubm90YXRpb25IABI+CgxyZWNvcmRfdmFsdWUYWCABKAsyJi50dXJiby5yZXN1bHQudjEuUmVjb3JkVmFsdWVBbm5vdGF0aW9uSAASRAoPZW50aXR5X3NhbGllbmNlGFkgASgLMikudHVyYm8ucmVzdWx0LnYxLkVudGl0eVNhbGllbmNlQW5ub3RhdGlvbkgAEkkKEmNodW5rX3RhYmxlX2hlYWRlchhaIAEoCzIrLnR1cmJvLnJlc3VsdC52MS5DaHVua1RhYmxlSGVhZGVyQW5ub3RhdGlvbkgAEkEKDHRleHRfcXVhbGl0eRhbIAEoCzIpLnR1cmJvLnJlc3VsdC52MS5RdWFsaXR5TWV0cmljc0Fubm90YXRpb25IABJBCgxzcGFuX3F1YWxpdHkYXCABKAsyKS50dXJiby5yZXN1bHQudjEuUXVhbGl0eU1ldHJpY3NBbm5vdGF0aW9uSAASNwoIZXZpZGVuY2UYXSABKAsyIy50dXJiby5yZXN1bHQudjEuRXZpZGVuY2VBbm5vdGF0aW9uSAASOQoJcXVvdGF0aW9uGF4gASgLMiQudHVyYm8ucmVzdWx0LnYxLlF1b3RhdGlvbkFubm90YXRpb25IABI6CgpsYXlvdXRfYm94GF8gASgLMiQudHVyYm8ucmVzdWx0LnYxLkxheW91dEJveEFubm90YXRpb25IABJCCg5jaHVua19jaXRhdGlvbhhgIAEoCzIoLnR1cmJvLnJlc3VsdC52MS5DaHVua0NpdGF0aW9uQW5ub3RhdGlvbkgAEkoKEmdyYW1tYXJfZGlhZ25vc3RpYxhhIAEoCzIsLnR1cmJvLnJlc3VsdC52MS5HcmFtbWFyRGlhZ25vc3RpY0Fubm90YXRpb25IABI2CghwaWlfZWRpdBhiIAEoCzIiLnR1cmJvLnJlc3VsdC52MS5QaWlFZGl0QW5ub3RhdGlvbkgAEkAKDXBpaV9jYW5kaWRhdGUYYyABKAsyJy50dXJiby5yZXN1bHQudjEuUGlpQ2FuZGlkYXRlQW5ub3RhdGlvbkgAEjYKCHBpaV9yaXNrGGQgASgLMiIudHVyYm8ucmVzdWx0LnYxLlBpaVJpc2tBbm5vdGF0aW9uSABCBwoFdmFsdWVCDgoMX3Byb2JhYmlsaXR5QgwKCl9tb2RlbF9yZWYqVQoKT2Zmc2V0TW9kZRIUChBPRkZTRVRfTU9ERV9VVEY4EAASFQoRT0ZGU0VUX01PREVfVVRGMTYQARIaChZPRkZTRVRfTU9ERV9DT0RFX1BPSU5UEAIqiwEKD0xheWVyV2lyZVN0YXR1cxIhCh1MQVlFUl9XSVJFX1NUQVRVU19VTlNQRUNJRklFRBAAEhgKFExBWUVSX1dJUkVfU1RBVFVTX09LEAESHQoZTEFZRVJfV0lSRV9TVEFUVVNfU0tJUFBFRBACEhwKGExBWUVSX1dJUkVfU1RBVFVTX0ZBSUxFRBADKvcDCg5MYXllckVycm9yS2luZBIgChxMQVlFUl9FUlJPUl9LSU5EX1VOU1BFQ0lGSUVEEAASHQoZTEFZRVJfRVJST1JfS0lORF9ERUNMSU5FRBABEh0KGUxBWUVSX0VSUk9SX0tJTkRfVVBTVFJFQU0QAhIcChhMQVlFUl9FUlJPUl9LSU5EX01JU1NJTkcQAxIeChpMQVlFUl9FUlJPUl9LSU5EX0RVUExJQ0FURRAEEhoKFkxBWUVSX0VSUk9SX0tJTkRfQ0xBU0gQBRIbChdMQVlFUl9FUlJPUl9LSU5EX0JPVU5EUxAGEh8KG0xBWUVSX0VSUk9SX0tJTkRfTUlTQUxJR05FRBAHEh4KGkxBWUVSX0VSUk9SX0tJTkRfVU5BTElHTkVEEAgSHAoYTEFZRVJfRVJST1JfS0lORF9JTlZBTElEEAkSHQoZTEFZRVJfRVJST1JfS0lORF9QSVBFTElORRAKEh4KGkxBWUVSX0VSUk9SX0tJTkRfQU5OT1RBVE9SEAsSGgoWTEFZRVJfRVJST1JfS0lORF9QQU5JQxAMEhsKF0xBWUVSX0VSUk9SX0tJTkRfT0ZGU0VUEA0SGwoXTEFZRVJfRVJST1JfS0lORF9TT1VSQ0UQDhIaChZMQVlFUl9FUlJPUl9LSU5EX0xJTUlUEA8q1wEKEFByb2Nlc3NpbmdTdGF0dXMSIQodUFJPQ0VTU0lOR19TVEFUVVNfVU5TUEVDSUZJRUQQABIeChpQUk9DRVNTSU5HX1NUQVRVU19DT01QTEVURRABEh0KGVBST0NFU1NJTkdfU1RBVFVTX1BBUlRJQUwQAhIiCh5QUk9DRVNTSU5HX1NUQVRVU19OT1RfQVNTRVNTRUQQAxIcChhQUk9DRVNTSU5HX1NUQVRVU19GQUlMRUQQBBIfChtQUk9DRVNTSU5HX1NUQVRVU19DQU5DRUxMRUQQBSq7AQoQVHJ1bmNhdGlvblJlYXNvbhIhCh1UUlVOQ0FUSU9OX1JFQVNPTl9VTlNQRUNJRklFRBAAEiIKHlRSVU5DQVRJT05fUkVBU09OX01BWF9TRVFVRU5DRRABEh8KG1RSVU5DQVRJT05fUkVBU09OX01BWF9JVEVNUxACEh8KG1RSVU5DQVRJT05fUkVBU09OX01BWF9CWVRFUxADEh4KGlRSVU5DQVRJT05fUkVBU09OX0RFQURMSU5FEAQqVQoKQW5jaG9yS2luZBIUChBBTkNIT1JfS0lORF9URVhUEAASFwoTQU5DSE9SX0tJTkRfQUxJR05FRBABEhgKFEFOQ0hPUl9LSU5EX0RPQ1VNRU5UEAIqdwoLU3Bhbk91dGNvbWUSHAoYU1BBTl9PVVRDT01FX1VOU1BFQ0lGSUVEEAASFgoSU1BBTl9PVVRDT01FX0VYQUNUEAESGAoUU1BBTl9PVVRDT01FX1dJREVORUQQAhIYChRTUEFOX09VVENPTUVfREVMRVRFRBADKqcJCglQaWlLaW5kSWQSGwoXUElJX0tJTkRfSURfVU5TUEVDSUZJRUQQABITCg9QSUlfS0lORF9JRF9KV1QQARIeChpQSUlfS0lORF9JRF9BV1NfQUNDRVNTX0tFWRACEhwKGFBJSV9LSU5EX0lEX0dJVEhVQl9UT0tFThADEh4KGlBJSV9LSU5EX0lEX1VSTF9DUkVERU5USUFMEAQSFQoRUElJX0tJTkRfSURfRU1BSUwQBRIUChBQSUlfS0lORF9JRF9JQkFOEAYSFAoQUElJX0tJTkRfSURfSU1FSRAHEhQKEFBJSV9LSU5EX0lEX0NBUkQQCBIbChdQSUlfS0lORF9JRF9CVENfQUREUkVTUxAJEhsKF1BJSV9LSU5EX0lEX0VUSF9BRERSRVNTEAoSEwoPUElJX0tJTkRfSURfTUFDEAsSFAoQUElJX0tJTkRfSURfSVBWNhAMEhQKEFBJSV9LSU5EX0lEX0lQVjQQDRIWChJQSUlfS0lORF9JRF9VU19TU04QDhIXChNQSUlfS0lORF9JRF9VU19JVElOEA8SFgoSUElJX0tJTkRfSURfVUtfTkhTEBASHAoYUElJX0tJTkRfSURfREVfU1RFVUVSX0lEEBESFgoSUElJX0tJTkRfSURfRVNfRE5JEBISFgoSUElJX0tJTkRfSURfRVNfTklFEBMSFgoSUElJX0tJTkRfSURfRVNfQ0lGEBQSIgoeUElJX0tJTkRfSURfREVfUEVSU09OQUxBVVNXRUlTEBUSFgoSUElJX0tJTkRfSURfRlJfTklSEBYSGAoUUElJX0tJTkRfSURfRlJfU0lSRU4QFxIYChRQSUlfS0lORF9JRF9GUl9TSVJFVBAYEhYKElBJSV9LSU5EX0lEX0JSX0NQRhAZEhcKE1BJSV9LSU5EX0lEX0JSX0NOUEoQGhITCg9QSUlfS0lORF9JRF9CSUMQGxIWChJQSUlfS0lORF9JRF9DQV9TSU4QHBIbChdQSUlfS0lORF9JRF9BQkFfUk9VVElORxAdEhUKEVBJSV9LSU5EX0lEX1BIT05FEB4SEwoPUElJX0tJTkRfSURfVklOEB8SHwobUElJX0tJTkRfSURfVFJBQ0tJTkdfTlVNQkVSECASFgoSUElJX0tJTkRfSURfVVNfRUlOECESEwoPUElJX0tJTkRfSURfVVJMECISGwoXUElJX0tJTkRfSURfVVNfUEFTU1BPUlQQIxIiCh5QSUlfS0lORF9JRF9VU19EUklWRVJTX0xJQ0VOQ0UQJBIWChJQSUlfS0lORF9JRF9VU19aSVAQJRIfChtQSUlfS0lORF9JRF9QSE9ORV9FWFRFTlNJT04QJhITCg9QSUlfS0lORF9JRF9BR0UQJxIeChpQSUlfS0lORF9JRF9TVFJFRVRfQUREUkVTUxAoEhQKEFBJSV9LSU5EX0lEX0RBVEUQKRIYChRQSUlfS0lORF9JRF9DVVJSRU5DWRAqEhUKEVBJSV9LSU5EX0lEX1NUQVRFECsSFgoSUElJX0tJTkRfSURfQ09VTlRZECwSFAoQUElJX0tJTkRfSURfQ0lUWRAtKscBChFQaWlDb25maWRlbmNlVGllchIjCh9QSUlfQ09ORklERU5DRV9USUVSX1VOU1BFQ0lGSUVEEAASHgoaUElJX0NPTkZJREVOQ0VfVElFUl9GT1JNQVQQARIgChxQSUlfQ09ORklERU5DRV9USUVSX0NIRUNLU1VNEAISKQolUElJX0NPTkZJREVOQ0VfVElFUl9MQUJFTExFRF9DSEVDS1NVTRADEiAKHFBJSV9DT05GSURFTkNFX1RJRVJfTEFCRUxMRUQQBCqvBQoUUGlpQ2hlY2tzdW1BbGdvcml0aG0SJgoiUElJX0NIRUNLU1VNX0FMR09SSVRITV9VTlNQRUNJRklFRBAAEh8KG1BJSV9DSEVDS1NVTV9BTEdPUklUSE1fTFVIThABEiAKHFBJSV9DSEVDS1NVTV9BTEdPUklUSE1fTU9EOTcQAhIgChxQSUlfQ0hFQ0tTVU1fQUxHT1JJVEhNX01PRDExEAMSIAocUElJX0NIRUNLU1VNX0FMR09SSVRITV9NT0QyMxAEEiEKHVBJSV9DSEVDS1NVTV9BTEdPUklUSE1fRVNfQ0lGEAUSKwonUElJX0NIRUNLU1VNX0FMR09SSVRITV9JU083MDY0X01PRDExXzEwEAYSIwofUElJX0NIRUNLU1VNX0FMR09SSVRITV9JQ0FPOTMwMxAHEiYKIlBJSV9DSEVDS1NVTV9BTEdPUklUSE1fQkFTRTU4Q0hFQ0sQCBIhCh1QSUlfQ0hFQ0tTVU1fQUxHT1JJVEhNX0JFQ0gzMhAJEiAKHFBJSV9DSEVDS1NVTV9BTEdPUklUSE1fRUlQNTUQChImCiJQSUlfQ0hFQ0tTVU1fQUxHT1JJVEhNX0FCQV9XRUlHSFRTEAsSIAocUElJX0NIRUNLU1VNX0FMR09SSVRITV9PVEhFUhAMEigKJFBJSV9DSEVDS1NVTV9BTEdPUklUSE1fTEFfUE9TVEVfTU9ENRANEh4KGlBJSV9DSEVDS1NVTV9BTEdPUklUSE1fVklOEA4SJAogUElJX0NIRUNLU1VNX0FMR09SSVRITV9VUFNfTU9EMTAQDxImCiJQSUlfQ0hFQ0tTVU1fQUxHT1JJVEhNX0ZFREVYX01PRDExEBASJAogUElJX0NIRUNLU1VNX0FMR09SSVRITV9HUzFfTU9EMTAQESqGAQoMUGlpU2NvcmVLaW5kEh4KGlBJSV9TQ09SRV9LSU5EX1VOU1BFQ0lGSUVEEAASHgoaUElJX1NDT1JFX0tJTkRfUFJPQkFCSUxJVFkQARIYChRQSUlfU0NPUkVfS0lORF9MT0dJVBACEhwKGFBJSV9TQ09SRV9LSU5EX0hFVVJJU1RJQxADKrMBChFQaWlXaXRoaGVsZFJlYXNvbhIjCh9QSUlfV0lUSEhFTERfUkVBU09OX1VOU1BFQ0lGSUVEEAASJQohUElJX1dJVEhIRUxEX1JFQVNPTl9OT1RfUkVRVUVTVEVEEAESIwofUElJX1dJVEhIRUxEX1JFQVNPTl9OT1RfR1JBTlRFRBACEi0KKVBJSV9XSVRISEVMRF9SRUFTT05fTk9UX0FMTE9XRURfQllfUE9MSUNZEAMqngEKDFBpaVZhbHVlRm9ybRIeChpQSUlfVkFMVUVfRk9STV9VTlNQRUNJRklFRBAAEhsKF1BJSV9WQUxVRV9GT1JNX1dJVEhIRUxEEAESGAoUUElJX1ZBTFVFX0ZPUk1fVE9LRU4QAhIdChlQSUlfVkFMVUVfRk9STV9DSVBIRVJURVhUEAMSGAoUUElJX1ZBTFVFX0ZPUk1fQ0xFQVIQBCqiAQoPUGlpUmVqZWN0UmVhc29uEiEKHVBJSV9SRUpFQ1RfUkVBU09OX1VOU1BFQ0lGSUVEEAASJAogUElJX1JFSkVDVF9SRUFTT05fTEFCRUxfUkVRVUlSRUQQARIfChtQSUlfUkVKRUNUX1JFQVNPTl9USFJFU0hPTEQQAhIlCiFQSUlfUkVKRUNUX1JFQVNPTl9SVUxFX1BSRUNFREVOQ0UQAypmCg1QaWlBbGxvd1Njb3BlEh8KG1BJSV9BTExPV19TQ09QRV9VTlNQRUNJRklFRBAAEhoKFlBJSV9BTExPV19TQ09QRV9HTE9CQUwQARIYChRQSUlfQUxMT1dfU0NPUEVfS0lORBACKoIECgtQaWlTdHJhdGVneRIcChhQSUlfU1RSQVRFR1lfVU5TUEVDSUZJRUQQABIVChFQSUlfU1RSQVRFR1lfTUFTSxABEhoKFlBJSV9TVFJBVEVHWV9QU0VVRE9OWU0QAhIVChFQSUlfU1RSQVRFR1lfSE1BQxADEhcKE1BJSV9TVFJBVEVHWV9GT1JNQVQQBBIXChNQSUlfU1RSQVRFR1lfU1RBVElDEAUSFwoTUElJX1NUUkFURUdZX1NIQTI1NhAGEhQKEFBJSV9TVFJBVEVHWV9NQVAQBxIZChVQSUlfU1RSQVRFR1lfVFJVTkNBVEUQCBIbChdQSUlfU1RSQVRFR1lfQUJCUkVWSUFURRAJEhwKGFBJSV9TVFJBVEVHWV9BRVNfMjU2X0dDTRAKEhQKEFBJSV9TVFJBVEVHWV9GRjEQCxIgChxQSUlfU1RSQVRFR1lfWklQX1NBRkVfSEFSQk9SEAwSIAocUElJX1NUUkFURUdZX0FHRV9TQUZFX0hBUkJPUhANEhoKFlBJSV9TVFJBVEVHWV9EQVRFX1lFQVIQDhIbChdQSUlfU1RSQVRFR1lfREFURV9TSElGVBAPEh4KGlBJSV9TVFJBVEVHWV9EQVRFX1JFTEFUSVZFEBASIQodUElJX1NUUkFURUdZX0RBVEVfU0FGRV9IQVJCT1IQESqwAQoMUGlpUmlza0xldmVsEh4KGlBJSV9SSVNLX0xFVkVMX1VOU1BFQ0lGSUVEEAASFwoTUElJX1JJU0tfTEVWRUxfTk9ORRABEhYKElBJSV9SSVNLX0xFVkVMX0xPVxACEhkKFVBJSV9SSVNLX0xFVkVMX01FRElVTRADEhcKE1BJSV9SSVNLX0xFVkVMX0hJR0gQBBIbChdQSUlfUklTS19MRVZFTF9DUklUSUNBTBAFKowBCgtQaWlTZXZlcml0eRIcChhQSUlfU0VWRVJJVFlfVU5TUEVDSUZJRUQQABIUChBQSUlfU0VWRVJJVFlfTE9XEAESFwoTUElJX1NFVkVSSVRZX01FRElVTRACEhUKEVBJSV9TRVZFUklUWV9ISUdIEAMSGQoVUElJX1NFVkVSSVRZX0NSSVRJQ0FMEAQq0wEKE1BpaVN1Z2dlc3RlZFJld3JpdGUSJQohUElJX1NVR0dFU1RFRF9SRVdSSVRFX1VOU1BFQ0lGSUVEEAASHgoaUElJX1NVR0dFU1RFRF9SRVdSSVRFX01BU0sQARIqCiZQSUlfU1VHR0VTVEVEX1JFV1JJVEVfTUFTS19LRUVQX0ZPUk1BVBACEh8KG1BJSV9TVUdHRVNURURfUkVXUklURV9MQUJFTBADEigKJFBJSV9TVUdHRVNURURfUkVXUklURV9LRUVQX0xBU1RfRk9VUhAEYgZwcm90bzM");
var TypedAnnotationSchema = /* @__PURE__ */ messageDesc(file_turbo_typed_annotations, 147);
var LayerWireStatus = /* @__PURE__ */ ((LayerWireStatus2) => {
  LayerWireStatus2[LayerWireStatus2["UNSPECIFIED"] = 0] = "UNSPECIFIED";
  LayerWireStatus2[LayerWireStatus2["OK"] = 1] = "OK";
  LayerWireStatus2[LayerWireStatus2["SKIPPED"] = 2] = "SKIPPED";
  LayerWireStatus2[LayerWireStatus2["FAILED"] = 3] = "FAILED";
  return LayerWireStatus2;
})(LayerWireStatus || {});
var LayerErrorKind = /* @__PURE__ */ ((LayerErrorKind2) => {
  LayerErrorKind2[LayerErrorKind2["UNSPECIFIED"] = 0] = "UNSPECIFIED";
  LayerErrorKind2[LayerErrorKind2["DECLINED"] = 1] = "DECLINED";
  LayerErrorKind2[LayerErrorKind2["UPSTREAM"] = 2] = "UPSTREAM";
  LayerErrorKind2[LayerErrorKind2["MISSING"] = 3] = "MISSING";
  LayerErrorKind2[LayerErrorKind2["DUPLICATE"] = 4] = "DUPLICATE";
  LayerErrorKind2[LayerErrorKind2["CLASH"] = 5] = "CLASH";
  LayerErrorKind2[LayerErrorKind2["BOUNDS"] = 6] = "BOUNDS";
  LayerErrorKind2[LayerErrorKind2["MISALIGNED"] = 7] = "MISALIGNED";
  LayerErrorKind2[LayerErrorKind2["UNALIGNED"] = 8] = "UNALIGNED";
  LayerErrorKind2[LayerErrorKind2["INVALID"] = 9] = "INVALID";
  LayerErrorKind2[LayerErrorKind2["PIPELINE"] = 10] = "PIPELINE";
  LayerErrorKind2[LayerErrorKind2["ANNOTATOR"] = 11] = "ANNOTATOR";
  LayerErrorKind2[LayerErrorKind2["PANIC"] = 12] = "PANIC";
  LayerErrorKind2[LayerErrorKind2["OFFSET"] = 13] = "OFFSET";
  LayerErrorKind2[LayerErrorKind2["SOURCE"] = 14] = "SOURCE";
  LayerErrorKind2[LayerErrorKind2["LIMIT"] = 15] = "LIMIT";
  return LayerErrorKind2;
})(LayerErrorKind || {});

// demo/web/src/keep.ts
var PREFIX = "turbo-demo.input.";
var TABS = ["embed", "semantic", "analyze", "search", "vocabulary"];
function storable(tag, type) {
  const t = tag.toLowerCase();
  if (t === "textarea" || t === "select") return true;
  if (t !== "input") return false;
  return !["password", "file", "hidden", "submit", "button", "reset", "image"].includes(type.toLowerCase());
}
function stored(type, value, checked) {
  return type === "checkbox" || type === "radio" ? checked ? "1" : "0" : value;
}
function recall(id) {
  try {
    return localStorage.getItem(PREFIX + id);
  } catch {
    return null;
  }
}
function remember(id, value) {
  if (resetting) return;
  try {
    localStorage.setItem(PREFIX + id, value);
  } catch {
  }
}
function forget(id) {
  try {
    localStorage.removeItem(PREFIX + id);
  } catch {
  }
}
function fields(panel) {
  return [...panel.querySelectorAll("input[id], textarea[id], select[id]")].filter((f) => storable(f.tagName, f.type) && !f.hasAttribute("data-unkept"));
}
function put(f, v) {
  if (f instanceof HTMLInputElement && (f.type === "checkbox" || f.type === "radio")) f.checked = v === "1";
  else if (f instanceof HTMLSelectElement) {
    if ([...f.options].some((o) => o.value === v)) f.value = v;
  } else f.value = v;
}
var restored = /* @__PURE__ */ new WeakSet();
function restore(select5) {
  if (restored.has(select5)) return;
  const v = select5.id ? recall(select5.id) : null;
  if (v === null || !select5.closest(TABS.map((t) => `#tab-${t}`).join(", "))) return;
  if (!offered(v, [...select5.options].map((o) => o.value))) return;
  select5.value = v;
  restored.add(select5);
}
function offered(v, values) {
  return v !== null && values.includes(v);
}
var resetting = false;
var resets = /* @__PURE__ */ new Map();
function onReset(tab, again) {
  resets.set(tab, [...resets.get(tab) ?? [], again]);
}
function reset(tab, panel) {
  resetting = true;
  try {
    for (const f of fields(panel)) {
      forget(f.id);
      if (f instanceof HTMLInputElement && (f.type === "checkbox" || f.type === "radio")) f.checked = f.defaultChecked;
      else if (f instanceof HTMLSelectElement) f.selectedIndex = Math.max(0, [...f.options].findIndex((o) => o.defaultSelected));
      else f.value = f.defaultValue;
      f.dispatchEvent(new Event("change", { bubbles: true }));
    }
    for (const again of resets.get(tab) ?? []) again();
  } finally {
    resetting = false;
  }
}
function init() {
  for (const tab of TABS) {
    const panel = document.getElementById(`tab-${tab}`);
    if (!panel) continue;
    for (const f of fields(panel)) {
      const v = recall(f.id);
      if (v !== null) put(f, v);
    }
    const keep = (ev) => {
      const f = ev.target;
      if (f.id && storable(f.tagName ?? "", f.type ?? "") && !f.hasAttribute("data-unkept")) remember(f.id, stored(f.type, f.value, f.checked === true));
    };
    panel.addEventListener("input", keep);
    panel.addEventListener("change", keep);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "reset";
    button.id = `${tab}-reset`;
    button.textContent = "Reset to sample";
    button.title = "Put this tab's inputs back to the sample and forget what was typed";
    button.addEventListener("click", () => reset(tab, panel));
    panel.querySelector(".box")?.append(button);
  }
}
function recallList(id) {
  const v = recall(id);
  return v === null ? null : v.split("\n").filter((x) => x.length);
}
function forgetOwn(id) {
  forget(id);
}

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
function stat(value, key3) {
  return el("div", { class: "stat" }, el("div", { class: "v", text: value }), el("div", { class: "k", text: key3 }));
}
function table(head, rows2, cls = "", none = "None.") {
  if (!rows2.length) return el("p", { class: "muted empty", text: none });
  return el(
    "table",
    { class: cls || void 0 },
    el("tr", {}, head.map((h) => el("th", { text: h }))),
    rows2.map((r) => el("tr", {}, r.map((c) => el("td", {}, c))))
  );
}
function options(select5, values, labels, none = "") {
  const keep = select5.value;
  if (!values.length && none) return void select5.replaceChildren(el("option", { value: "", text: none }));
  select5.replaceChildren(...values.map((v, i) => el("option", { value: v, text: labels ? labels[i] : v })));
  if (values.includes(keep)) select5.value = keep;
  restore(select5);
}
function failure(x) {
  if (!x.details) return [x.text];
  return [x.text, " ", el("details", { class: "why" }, el("summary", { text: "details" }), el("code", { text: x.details }))];
}
function say(box5, what, kind = "") {
  if (typeof what === "string") {
    box5.textContent = what;
    box5.className = kind ? `note ${kind}` : "note";
    box5.hidden = what === "";
    return;
  }
  box5.replaceChildren();
  append(box5, failure(what));
  box5.className = what.text === "Cancelled." ? "note" : "note error";
  box5.hidden = false;
}
function sending(box5, what) {
  const bar3 = el("progress", { max: "1", value: "0", "aria-label": "Upload progress" });
  const line = el("span", {});
  box5.replaceChildren(line, bar3);
  box5.className = "note busy upload";
  box5.hidden = false;
  return (sent, total) => {
    bar3.max = Math.max(1, total);
    bar3.value = sent;
    const share = total ? Math.floor(100 * sent / total) : 100;
    line.textContent = sent < total ? `Uploading ${what}: ${size(sent)} of ${size(total)} (${share}%)` : `Sent ${size(total)} (${what}). The server is working on it\u2026`;
  };
}
function failed(x) {
  return el("p", { class: "error" }, failure(x));
}
function heatKey(what) {
  return el("p", { class: "heat-key muted" }, what, " ", el("span", { class: "ramp", role: "img", "aria-label": "colour scale from low to high" }, "low ", el("i", {}), " high"));
}
function reveal(box5) {
  if (!box5 || !matchMedia("(max-width: 820px)").matches) return () => {
  };
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const show3 = () => box5.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "start" });
  let moved = false;
  for (const ev of ["wheel", "touchmove", "keydown"]) {
    addEventListener(ev, () => moved = true, { once: true, passive: true });
  }
  show3();
  return () => {
    if (!moved) show3();
  };
}
function count(n2, noun) {
  return `${n2} ${noun}${n2 === 1 ? "" : "s"}`;
}
function size(n2) {
  const v = Number(n2);
  if (v < 1024) return `${v} B`;
  if (v < 1 << 20) return `${(v / 1024).toFixed(1)} KiB`;
  if (v < 1 << 30) return `${(v / (1 << 20)).toFixed(1)} MiB`;
  return `${(v / (1 << 30)).toFixed(2)} GiB`;
}
function elapsed(ms) {
  if (ms < 6e4) return `${(ms / 1e3).toFixed(1)} s`;
  const s2 = Math.floor(ms / 1e3);
  return `${Math.floor(s2 / 60)} min ${String(s2 % 60).padStart(2, "0")} s`;
}
function download(name, data, type) {
  const url = URL.createObjectURL(new Blob(data, { type }));
  const a = el("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1e4);
}
function after(s2, prefix2) {
  return s2.startsWith(prefix2) ? s2.slice(prefix2.length) : s2;
}
function lazy(details, render6) {
  let stale2 = true;
  details.addEventListener("toggle", () => {
    if (details.open && stale2) {
      stale2 = false;
      render6();
    }
  });
  return () => {
    stale2 = true;
    if (details.open) {
      stale2 = false;
      render6();
    }
  };
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
  const [container, field3, get] = createExtensionContainer(extension);
  const ctx = makeReadContext(options2);
  for (const uf of ufs) {
    readField(container, new BinaryReader(uf.data), field3, uf.wireType, ctx);
  }
  return get();
}
function setExtension(message, extension, value) {
  var _a;
  assertExtendee(extension, message);
  const ufs = ((_a = message.$unknown) !== null && _a !== void 0 ? _a : []).filter((uf) => uf.no !== extension.number);
  const [container, field3] = createExtensionContainer(extension, value);
  const writer = new BinaryWriter();
  writeField(writer, { writeUnknownFields: true }, container, field3);
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
  const field3 = Object.assign(Object.assign({}, extension), { kind: "field", parent: extension.extendee, localName });
  const desc = Object.assign(Object.assign({}, extension.extendee), { fields: [field3], members: [field3], oneofs: [] });
  const container = create(desc, value !== void 0 ? { [localName]: value } : void 0);
  return [
    reflect(desc, container),
    field3,
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
  for (const field3 of sortedFields) {
    const writeField2 = compileField2(field3);
    fieldWriters.push(field3.jsonName === "__proto__" || field3.name === "__proto__" ? writeOwnKeys(writeField2) : writeField2);
  }
  return compiledWriter3;
}
function writeOwnKeys(writeField2) {
  return (opts, message, json) => {
    const tmp = /* @__PURE__ */ Object.create(null);
    writeField2(opts, message, tmp);
    for (const key3 of Object.keys(tmp)) {
      setOwn(json, key3, tmp[key3]);
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
function compileField2(field3) {
  switch (field3.fieldKind) {
    case "scalar":
    case "enum":
    case "message":
      return compileSingularField2(field3);
    case "list":
    case "map": {
      const writeValue = field3.fieldKind == "list" ? compileListValue(field3) : compileMapValue(field3);
      const protoName = field3.name;
      const jsonKey = field3.jsonName;
      const localName = field3.localName;
      return (opts, message, json) => {
        const value = writeValue(opts, message[localName]);
        if (value !== void 0) {
          json[opts.useProtoFieldName ? protoName : jsonKey] = value;
        }
      };
    }
  }
}
function compileSingularField2(field3) {
  const writeValue = compileSingularValue2(field3);
  const protoName = field3.name;
  const jsonKey = field3.jsonName;
  const localName = field3.localName;
  if (field3.oneof) {
    const oneofLocalName = field3.oneof.localName;
    return (opts, message, json) => {
      const oneof = message[oneofLocalName];
      if (oneof.case === localName) {
        json[opts.useProtoFieldName ? protoName : jsonKey] = writeValue(opts, oneof.value);
      }
    };
  }
  if (field3.presence != IMPLICIT5) {
    const requiredError = field3.presence == LEGACY_REQUIRED3 ? `cannot encode ${field3} to JSON: required field not set` : void 0;
    return (opts, message, json) => {
      const value = message[localName];
      if (value !== void 0 && Object.prototype.hasOwnProperty.call(message, localName)) {
        json[opts.useProtoFieldName ? protoName : jsonKey] = writeValue(opts, value);
      } else if (requiredError !== void 0) {
        throw new Error(requiredError);
      }
    };
  }
  if (field3.fieldKind == "enum") {
    const zero = field3.enum.values[0].number;
    return (opts, message, json) => {
      const value = message[localName];
      if (value !== zero || opts.alwaysEmitImplicit) {
        json[opts.useProtoFieldName ? protoName : jsonKey] = writeValue(opts, value);
      }
    };
  }
  switch (field3.scalar) {
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
function compileFieldValue(field3) {
  switch (field3.fieldKind) {
    case "scalar":
    case "enum":
    case "message":
      return compileSingularValue2(field3);
    case "list":
      return compileListValue(field3);
    case "map":
      return compileMapValue(field3);
  }
}
function compileSingularValue2(field3) {
  switch (field3.fieldKind) {
    case "scalar":
      return compileScalarValue2(field3);
    case "enum":
      return compileEnumValue(field3);
    case "message":
      return compileMessageValue(field3);
  }
}
function compileMessageValue(field3) {
  const { toMessage } = localMessageMapper(field3);
  const writeMessage = compiledWriter2(field3.message);
  return (opts, value) => writeMessage(opts, toMessage(value));
}
function compileListValue(field3) {
  const writeItem = compileListItemValue(field3);
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
function compileListItemValue(field3) {
  switch (field3.listKind) {
    case "scalar":
      return compileScalarValue2(field3);
    case "enum":
      return compileEnumValue(field3);
    case "message":
      return compileMessageValue(field3);
  }
}
function compileMapValue(field3) {
  const writeMapValue = compileMapEntryValue(field3);
  return (opts, value) => {
    const record = value;
    const keys = Object.keys(record);
    if (keys.length == 0 && !opts.alwaysEmitImplicit) {
      return void 0;
    }
    const jsonObject = {};
    for (let i = 0; i < keys.length; i++) {
      const key3 = keys[i];
      setOwn(jsonObject, key3, writeMapValue(opts, record[key3]));
    }
    return jsonObject;
  };
}
function compileMapEntryValue(field3) {
  switch (field3.mapKind) {
    case "scalar":
      return compileScalarValue2(field3);
    case "enum":
      return compileEnumValue(field3);
    case "message":
      return compileMessageValue(field3);
  }
}
function compileEnumValue(field3) {
  const desc = field3.enum;
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
function compileScalarValue2(field3) {
  switch (field3.scalar) {
    // int32, fixed32, uint32: JSON value will be a decimal number. Either numbers or strings are accepted.
    case ScalarType.INT32:
    case ScalarType.SFIXED32:
    case ScalarType.SINT32:
    case ScalarType.FIXED32:
    case ScalarType.UINT32:
      return (opts, value) => {
        if (typeof value != "number") {
          throw errorScalarValue(field3, value);
        }
        return value;
      };
    // float, double: JSON value will be a number or one of the special string values "NaN", "Infinity", and "-Infinity".
    // Either numbers or strings are accepted. Exponent notation is also accepted.
    case ScalarType.FLOAT:
    case ScalarType.DOUBLE:
      return (opts, value) => {
        if (typeof value != "number") {
          throw errorScalarValue(field3, value);
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
          throw errorScalarValue(field3, value);
        }
        return value;
      };
    // bool:
    case ScalarType.BOOL:
      return (opts, value) => {
        if (typeof value != "boolean") {
          throw errorScalarValue(field3, value);
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
        throw errorScalarValue(field3, value);
      };
    // bytes: JSON value will be the data encoded as a string using standard base64 encoding with paddings.
    // Either standard or URL-safe base64 encoding with/without paddings are accepted.
    case ScalarType.BYTES:
      return (opts, value) => {
        if (value instanceof Uint8Array) {
          return base64Encode(value);
        }
        throw errorScalarValue(field3, value);
      };
  }
}
function errorScalarValue(field3, value) {
  var _a;
  return new Error(`cannot encode ${field3} to JSON: ${(_a = checkField(field3, value)) === null || _a === void 0 ? void 0 : _a.message}`);
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
      const [container, field3] = createExtensionContainer(extension, value);
      const local = container[unsafeLocal];
      const jsonValue = compileFieldValue(field3)(opts, local[field3.localName]);
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
    const key3 = keys[i];
    setOwn(json, key3, valueToJson(val.fields[key3]));
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
        const field3 = entry.field;
        if (fieldSeen.has(field3)) {
          throw new FieldError(field3, "set multiple times");
        }
        fieldSeen.add(field3);
        if (entry.oneofScalarNullSkip && jsonValue === null) {
          continue;
        }
        if (entry.oneof) {
          const seen = oneofSeen.get(entry.oneof);
          if (seen !== void 0) {
            throw new FieldError(entry.oneof, `oneof set multiple times by ${seen.name} and ${field3.name}`);
          }
          oneofSeen.set(entry.oneof, field3);
        }
        entry.read(message, jsonValue, ctx);
      } else {
        const extension = jsonKey.startsWith("[") && jsonKey.endsWith("]") ? (_a = ctx.registry) === null || _a === void 0 ? void 0 : _a.getExtension(jsonKey.substring(1, jsonKey.length - 1)) : void 0;
        if ((extension === null || extension === void 0 ? void 0 : extension.extendee.typeName) == typeName) {
          const [container, field3, get] = createExtensionContainer(extension);
          compileFieldReader2(field3)(container[unsafeLocal], jsonValue, ctx);
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
  for (const field3 of desc.fields) {
    const entry = {
      read: compileFieldReader2(field3),
      field: field3,
      oneof: field3.oneof,
      oneofScalarNullSkip: field3.oneof !== void 0 && field3.fieldKind == "scalar"
    };
    fieldsByJsonKey.set(field3.name, entry).set(field3.jsonName, entry);
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
        const scalar2 = valueField.scalar;
        const longAsString = valueField.longAsString;
        const readScalar = compileScalarConverter(valueField);
        return (message, json) => {
          if (json === null) {
            message[localName] = scalarZeroValue(scalar2, longAsString);
          } else {
            message[localName] = readScalar(json);
          }
        };
      }
      return void 0;
  }
}
function compileFieldReader2(field3) {
  switch (field3.fieldKind) {
    case "scalar":
      return compileScalarFieldReader2(field3);
    case "enum":
      return compileEnumFieldReader2(field3);
    case "message":
      return compileMessageFieldReader2(field3);
    case "list":
      return compileListFieldReader2(field3);
    case "map":
      return compileMapFieldReader2(field3);
  }
}
function compileScalarFieldReader2(field3) {
  const readScalar = compileScalarConverter(field3);
  const localName = field3.localName;
  if (field3.oneof) {
    const oneofLocalName = field3.oneof.localName;
    return (message, json) => {
      message[oneofLocalName] = {
        case: localName,
        value: readScalar(json)
      };
    };
  }
  const clear = compileClear(field3);
  return (message, json) => {
    if (json === null) {
      clear(message);
    } else {
      message[localName] = readScalar(json);
    }
  };
}
function compileClear(field3) {
  const localName = field3.localName;
  if (field3.presence != IMPLICIT6) {
    return (message) => {
      delete message[localName];
    };
  }
  if (field3.fieldKind == "enum") {
    const zero = field3.enum.values[0].number;
    return (message) => {
      message[localName] = zero;
    };
  }
  const scalar2 = field3.scalar;
  const longAsString = field3.longAsString;
  return (message) => {
    message[localName] = scalarZeroValue(scalar2, longAsString);
  };
}
function compileEnumFieldReader2(field3) {
  const readEnumValue = compileEnumConverter(field3.enum);
  const checkEnum = compileEnumCheck(field3.enum);
  const localName = field3.localName;
  const nullResets = field3.enum.typeName != "google.protobuf.NullValue";
  if (field3.oneof) {
    const oneofLocalName = field3.oneof.localName;
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
      const check2 = checkEnum(value);
      if (check2 !== true) {
        throw new FieldError(field3, reasonSingular(field3, value, check2));
      }
      message[oneofLocalName] = { case: localName, value };
    };
  }
  const clear = compileClear(field3);
  return (message, json, ctx) => {
    if (json === null && nullResets) {
      clear(message);
      return;
    }
    const value = readEnumValue(json, ctx.ignoreUnknownFields);
    if (value === tokenIgnoredUnknownEnum) {
      return;
    }
    const check2 = checkEnum(value);
    if (check2 !== true) {
      throw new FieldError(field3, reasonSingular(field3, value, check2));
    }
    message[localName] = value;
  };
}
function compileMessageFieldReader2(field3) {
  const localName = field3.localName;
  const { toMessage, toLocal } = localMessageMapper(field3);
  const readChild = compiledReader2(field3.message);
  const nullResets = field3.message.typeName != "google.protobuf.Value";
  if (field3.oneof) {
    const oneofLocalName = field3.oneof.localName;
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
function compileListFieldReader2(field3) {
  const localName = field3.localName;
  const readItem = compileListItemReader(field3);
  return (message, json, ctx) => {
    if (json === null) {
      return;
    }
    if (!Array.isArray(json)) {
      throw new FieldError(field3, "expected Array, got " + formatVal(json));
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
function compileListItemReader(field3) {
  switch (field3.listKind) {
    case "scalar": {
      const parseScalar = compileScalarParse(field3);
      const checkValue = checkScalarValue(field3.scalar);
      const toLocal = compileScalarToLocal(field3);
      return (json, ctx, index) => {
        if (json === null) {
          throw new FieldError(field3, "list item must not be null");
        }
        const value = parseScalar(json);
        const check2 = checkValue(value);
        if (check2 !== true) {
          throw new FieldError(field3, `list item #${index + 1}: ${reasonSingular(field3, value, check2)}`);
        }
        return toLocal(value);
      };
    }
    case "enum": {
      const readEnumValue = compileEnumConverter(field3.enum);
      const checkEnum = compileEnumCheck(field3.enum);
      const nullResets = field3.enum.typeName != "google.protobuf.NullValue";
      return (json, ctx, index) => {
        if (json === null && nullResets) {
          throw new FieldError(field3, "list item must not be null");
        }
        const value = readEnumValue(json, ctx.ignoreUnknownFields);
        if (value === tokenIgnoredUnknownEnum) {
          return value;
        }
        const check2 = checkEnum(value);
        if (check2 !== true) {
          throw new FieldError(field3, `list item #${index + 1}: ${reasonSingular(field3, value, check2)}`);
        }
        return value;
      };
    }
    case "message": {
      const { toMessage, toLocal } = localMessageMapper(field3);
      const readChild = compiledReader2(field3.message);
      const nullResets = field3.message.typeName != "google.protobuf.Value";
      return (json, ctx) => {
        if (json === null && nullResets) {
          throw new FieldError(field3, "list item must not be null");
        }
        const child = toMessage(void 0);
        readChild(child, json, ctx);
        return toLocal(child);
      };
    }
  }
}
function compileMapFieldReader2(field3) {
  const localName = field3.localName;
  const mapKey = field3.mapKey;
  const parseMapKey = compileMapKeyParse(mapKey);
  const checkMapKey = checkScalarValue(mapKey);
  let parseValue;
  let checkValue;
  let toLocalValue = (value) => value;
  let nullResets = true;
  switch (field3.mapKind) {
    case "scalar": {
      parseValue = compileScalarParse(field3);
      checkValue = checkScalarValue(field3.scalar);
      toLocalValue = compileScalarToLocal(field3);
      break;
    }
    case "enum": {
      const readEnumValue = compileEnumConverter(field3.enum);
      parseValue = (json, ctx) => readEnumValue(json, ctx.ignoreUnknownFields);
      checkValue = compileEnumCheck(field3.enum);
      nullResets = field3.enum.typeName != "google.protobuf.NullValue";
      break;
    }
    case "message": {
      const { toMessage, toLocal } = localMessageMapper(field3);
      const readChild = compiledReader2(field3.message);
      nullResets = field3.message.typeName != "google.protobuf.Value";
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
      throw new FieldError(field3, "expected object, got " + formatVal(json));
    }
    const record = message[localName];
    const seen = /* @__PURE__ */ new Set();
    const jsonMapKeys = Object.keys(json);
    for (let i = 0; i < jsonMapKeys.length; i++) {
      const jsonMapKey = jsonMapKeys[i];
      const jsonMapValue = json[jsonMapKey];
      const key3 = parseMapKey(jsonMapKey);
      if (seen.has(key3)) {
        throw new FieldError(field3, `duplicate map key "${jsonMapKey}"`);
      }
      seen.add(key3);
      if (jsonMapValue === null && nullResets) {
        throw new FieldError(field3, "map value must not be null");
      }
      const value = parseValue(jsonMapValue, ctx);
      if (value === tokenIgnoredUnknownEnum) {
        continue;
      }
      const checkKey = checkMapKey(key3);
      if (checkKey !== true) {
        throw new FieldError(field3, `invalid map key: ${reasonSingular({ scalar: mapKey }, key3, checkKey)}`);
      }
      if (checkValue !== void 0) {
        const check2 = checkValue(value);
        if (check2 !== true) {
          throw new FieldError(field3, `map entry ${formatVal(key3)}: ${reasonSingular(field3, value, check2)}`);
        }
      }
      setOwn(record, key3, toLocalValue(value));
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
function compileScalarConverter(field3) {
  const parseScalar = compileScalarParse(field3);
  const checkValue = checkScalarValue(field3.scalar);
  const toLocal = compileScalarToLocal(field3);
  return (json) => {
    const value = parseScalar(json);
    const check2 = checkValue(value);
    if (check2 !== true) {
      throw new FieldError(field3, reasonSingular(field3, value, check2));
    }
    return toLocal(value);
  };
}
function compileScalarParse(field3) {
  switch (field3.scalar) {
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
            throw new FieldError(field3, "unexpected NaN number");
          }
          if (!Number.isFinite(json)) {
            throw new FieldError(field3, "unexpected infinite number");
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
            throw new FieldError(field3, message);
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
function compileScalarToLocal(field3) {
  const longAsString = field3.fieldKind !== "map" && field3.longAsString;
  switch (field3.scalar) {
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
        const open2 = i++;
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
          const name = escaped ? JSON.parse(jsonString.substring(open2, close + 1)) : jsonString.substring(open2 + 1, close);
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
    const copy2 = {};
    for (const key3 of Object.keys(json)) {
      if (key3 !== "@type") {
        setOwn(copy2, key3, json[key3]);
      }
    }
    compiledReader2(desc)(message, copy2, ctx);
  }
  anyPack(desc, message, any);
}
function timestampFromJson(timestamp, json) {
  if (typeof json !== "string") {
    throw new Error(`cannot decode message ${timestamp.$typeName} from JSON: ${formatVal(json)}`);
  }
  const matches2 = json.match(/^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,9}))?(?:Z|([+-][0-9][0-9]:[0-9][0-9]))$/);
  if (!matches2) {
    throw new Error(`cannot decode message ${timestamp.$typeName} from JSON: invalid RFC 3339 string`);
  }
  const ms = Date.parse(
    // biome-ignore format: want this to read well
    matches2[1] + "-" + matches2[2] + "-" + matches2[3] + "T" + matches2[4] + ":" + matches2[5] + ":" + matches2[6] + (matches2[8] ? matches2[8] : "Z")
  );
  if (Number.isNaN(ms)) {
    throw new Error(`cannot decode message ${timestamp.$typeName} from JSON: invalid RFC 3339 string`);
  }
  if (ms < timestampMsMin || ms > timestampMsMax) {
    throw new Error(`cannot decode message ${timestamp.$typeName} from JSON: must be from 0001-01-01T00:00:00Z to 9999-12-31T23:59:59Z inclusive`);
  }
  timestamp.seconds = protoInt64.parse(ms / 1e3);
  timestamp.nanos = 0;
  if (matches2[7]) {
    timestamp.nanos = parseInt("1" + matches2[7] + "0".repeat(9 - matches2[7].length)) - 1e9;
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
  fieldMask.paths = json.split(",").map((path2) => {
    if (path2.includes("_")) {
      throw new Error(`cannot decode message ${fieldMask.$typeName} from JSON: path names must be lowerCamelCase`);
    }
    return protoSnakeCase(path2);
  });
}
function structFromJson(struct, json, ctx) {
  if (typeof json != "object" || json == null || Array.isArray(json)) {
    throw new Error(`cannot decode message ${struct.$typeName} from JSON ${formatVal(json)}`);
  }
  const keys = Object.keys(json);
  for (let i = 0; i < keys.length; i++) {
    const key3 = keys[i];
    const parsedValue = create(ValueSchema);
    valueFromJson(parsedValue, json[key3], ctx);
    setOwn(struct.fields, key3, parsedValue);
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
    e.forEach((value, key3) => {
      h.append(key3, value);
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
        const result2 = await reader.read();
        if (result2.done) {
          if (buffer.byteLength > 0) {
            controller.error(new ConnectError("protocol error: incomplete envelope", Code.InvalidArgument));
          }
          controller.close();
        } else {
          for (const env of buffer.decode(result2.value)) {
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
  function verb(n2) {
    i[n2] = o[n2] && function(v) {
      return new Promise(function(resolve, reject) {
        v = o[n2](v), settle2(resolve, reject, v.done, v.value);
      });
    };
  }
  function settle2(resolve, reject, d, v) {
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
  function verb(n2, f) {
    if (g[n2]) {
      i[n2] = function(v) {
        return new Promise(function(a, b) {
          q.push([n2, v, a, b]) > 1 || resume(n2, v);
        });
      };
      if (f) i[n2] = f(i[n2]);
    }
  }
  function resume(n2, v) {
    try {
      step(g[n2](v));
    } catch (e) {
      settle2(q[0][3], e);
    }
  }
  function step(r) {
    r.value instanceof __await ? Promise.resolve(r.value.v).then(fulfill, reject) : settle2(q[0][2], r);
  }
  function fulfill(value) {
    resume("next", value);
  }
  function reject(value) {
    resume("throw", value);
  }
  function settle2(f, v) {
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
  function verb(n2, f) {
    i[n2] = o[n2] ? function(v) {
      return (p = !p) ? { value: __await(o[n2](v)), done: false } : f ? f(v) : v;
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
  function verb(n2) {
    i[n2] = o[n2] && function(v) {
      return new Promise(function(resolve, reject) {
        v = o[n2](v), settle2(resolve, reject, v.done, v.value);
      });
    };
  }
  function settle2(resolve, reject, d, v) {
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
  function verb(n2, f) {
    i[n2] = o[n2] ? function(v) {
      return (p = !p) ? { value: __await2(o[n2](v)), done: false } : f ? f(v) : v;
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
  function verb(n2, f) {
    if (g[n2]) {
      i[n2] = function(v) {
        return new Promise(function(a, b) {
          q.push([n2, v, a, b]) > 1 || resume(n2, v);
        });
      };
      if (f) i[n2] = f(i[n2]);
    }
  }
  function resume(n2, v) {
    try {
      step(g[n2](v));
    } catch (e) {
      settle2(q[0][3], e);
    }
  }
  function step(r) {
    r.value instanceof __await2 ? Promise.resolve(r.value.v).then(fulfill, reject) : settle2(q[0][2], r);
  }
  function fulfill(value) {
    resume("next", value);
  }
  function reject(value) {
    resume("throw", value);
  }
  function settle2(f, v) {
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
    let count2 = 0;
    try {
      for (var _f = true, _g = __asyncValues2(response.message), _h; _h = await _g.next(), _a = _h.done, !_a; _f = true) {
        _c = _h.value;
        _f = false;
        const message = _c;
        singleMessage = message;
        count2++;
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
    if (count2 > 1) {
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
  const sa = signals.filter((s2) => s2 !== void 0).concat(controller.signal);
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
    get(key3) {
      return key3.id in this ? this[key3.id] : key3.defaultValue;
    },
    set(key3, value) {
      this[key3.id] = value;
      return this;
    },
    delete(key3) {
      delete this[key3.id];
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
  function transform(result2) {
    if (result2.done === true) {
      return result2;
    }
    return {
      done: result2.done,
      value: normalize(desc, result2.value)
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
  const output2 = useBinaryFormat ? createBinarySerialization(method.output, binaryOptions) : createJsonSerialization(method.output, jsonOptions);
  return { parse: output2.parse, serialize: input2.serialize };
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
    new Headers(metadata).forEach((value, key3) => fallback.metadata.append(key3, value));
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
  const error2 = new ConnectError(message !== null && message !== void 0 ? message : "", code, metadata);
  error2.isWireError = true;
  if ("details" in jsonValue && Array.isArray(jsonValue.details)) {
    for (const detail of jsonValue.details) {
      if (detail === null || typeof detail != "object" || Array.isArray(detail) || typeof detail.type != "string" || typeof detail.value != "string") {
        throw fallback;
      }
      try {
        error2.details.push({
          type: detail.type,
          value: base64Decode(detail.value),
          debug: detail.debug
        });
      } catch (e) {
        throw fallback;
      }
    }
  }
  return error2;
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
    for (const [key3, values] of Object.entries(jsonValue.metadata)) {
      if (!Array.isArray(values) || values.some((value) => typeof value != "string")) {
        throw parseErr;
      }
      for (const value of values) {
        metadata.append(key3, value);
      }
    }
  }
  const error2 = "error" in jsonValue && jsonValue.error != null ? errorFromJson(jsonValue.error, metadata, parseErr) : void 0;
  return { metadata, error: error2 };
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
  header.forEach((value, key3) => {
    if (key3.toLowerCase().startsWith("trailer-")) {
      t.append(key3.substring(8), value);
    } else {
      h.append(key3, value);
    }
  });
  return [h, t];
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/version.js
var protocolVersion = "1";

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/request-header.js
function requestHeader(methodKind, useBinaryFormat, timeoutMs, userProvidedHeaders, setUserAgent) {
  const result2 = new Headers(userProvidedHeaders !== null && userProvidedHeaders !== void 0 ? userProvidedHeaders : {});
  if (timeoutMs !== void 0) {
    result2.set(headerTimeout, `${timeoutMs}`);
  }
  result2.set(headerContentType, methodKind == "unary" ? useBinaryFormat ? contentTypeUnaryProto : contentTypeUnaryJson : useBinaryFormat ? contentTypeStreamProto : contentTypeStreamJson);
  result2.set(headerProtocolVersion, protocolVersion);
  if (!result2.has(headerUserAgent) && setUserAgent) {
    result2.set(headerUserAgent, "connect-es/2.2.0");
  }
  return result2;
}

// demo/web/node_modules/@connectrpc/connect/dist/esm/protocol-connect/validate-response.js
function validateResponse(methodKind, useBinaryFormat, status3, headers) {
  const mimeType = headers.get(headerContentType);
  const parsedType = parseContentType(mimeType);
  if (status3 !== 200) {
    const errorFromStatus = new ConnectError(`HTTP ${status3}`, codeFromHttpStatus(status3), headers);
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
  function verb(n2, f) {
    if (g[n2]) {
      i[n2] = function(v) {
        return new Promise(function(a, b) {
          q.push([n2, v, a, b]) > 1 || resume(n2, v);
        });
      };
      if (f) i[n2] = f(i[n2]);
    }
  }
  function resume(n2, v) {
    try {
      step(g[n2](v));
    } catch (e) {
      settle2(q[0][3], e);
    }
  }
  function step(r) {
    r.value instanceof __await3 ? Promise.resolve(r.value.v).then(fulfill, reject) : settle2(q[0][2], r);
  }
  function fulfill(value) {
    resume("next", value);
  }
  function reject(value) {
    resume("throw", value);
  }
  function settle2(f, v) {
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
          const fetch = (_a2 = options2.fetch) !== null && _a2 !== void 0 ? _a2 : globalThis.fetch;
          const response = await fetch(req.url, Object.assign(Object.assign({}, fetchOptions), { method: req.requestMethod, headers: req.header, signal: req.signal, body: body2 }));
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
            const result2 = yield __await3(reader.read());
            if (result2.done) {
              break;
            }
            const { flags, data } = result2.value;
            if ((flags & compressedFlag) === compressedFlag) {
              throw new ConnectError(`protocol error: received unsupported compressed output`, Code.Internal);
            }
            if ((flags & endStreamFlag) === endStreamFlag) {
              endStreamReceived = true;
              const endStream = endStreamFromJson(data);
              if (endStream.error) {
                const error2 = endStream.error;
                header2.forEach((value, key3) => {
                  error2.metadata.append(key3, value);
                });
                throw error2;
              }
              endStream.metadata.forEach((value, key3) => trailerTarget.set(key3, value));
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
          const fetch = (_a2 = options2.fetch) !== null && _a2 !== void 0 ? _a2 : globalThis.fetch;
          const fRes = await fetch(req.url, Object.assign(Object.assign({}, fetchOptions), { method: req.requestMethod, headers: req.header, signal: req.signal, body: await createRequestBody(req.message) }));
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

// demo/web/src/errors.ts
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
function cancelled(e) {
  if (e instanceof Error && e.name === "AbortError") return true;
  return e instanceof ConnectError && e.code === Code.Canceled;
}
var EMPTY = {
  index_name: "Pick or create an index first.",
  natural_query: "Enter something to search for.",
  query: "Enter a query.",
  text: "Enter some text first.",
  name: "Enter a name.",
  bundle: "Give the bundle's path on the server.",
  collection_name: "Name the collection.",
  teacher: "Pick a teacher model.",
  model: "Pick a model.",
  model_name: "Pick a model.",
  artifact_id: "Pick an artifact.",
  graph: "Analyze a document first.",
  labels: "Enter at least one label."
};
var EMPTY_ENDS = [" must not be empty", " is required", " is empty", " must be set"];
function unprefixed(message) {
  const at = message.indexOf(": ");
  if (at < 0 || !message.startsWith("TURBO_E_")) return message;
  for (const ch of message.slice(0, at)) {
    const ok = ch === "_" || ch >= "A" && ch <= "Z" || ch >= "0" && ch <= "9";
    if (!ok) return message;
  }
  return message.slice(at + 2);
}
function sentence(message) {
  const m = message.trim();
  if (!m) return "";
  const first = String.fromCodePoint(m.codePointAt(0));
  const rest = m.slice(first.length);
  const last2 = m[m.length - 1];
  return first.toUpperCase() + rest + (last2 === "." || last2 === "!" || last2 === "?" ? "" : ".");
}
function emptyField(message) {
  for (const end of EMPTY_ENDS) {
    if (!message.endsWith(end)) continue;
    const field3 = message.slice(0, message.length - end.length);
    return EMPTY[field3.slice(field3.lastIndexOf(".") + 1)] ?? "";
  }
  return "";
}
function plain(code, message) {
  switch (code) {
    case Code.PermissionDenied: {
      const at = message.indexOf(" is not on the demo's allowlist");
      if (at > 0) return `The demo does not forward this write. Start it with --allow ${message.slice(message.startsWith("/") ? 1 : 0, at)}`;
      if (message.includes("without --user")) return "This acts for a user. Start the demo with --user NAME.";
      return message ? `Not allowed: ${sentence(message)}` : "Not allowed.";
    }
    case Code.Unauthenticated:
      return "The demo wants its bearer token. Enter it at the top of the page.";
    case Code.Unimplemented:
      return message ? `Not served: ${sentence(message)}` : "The server does not serve this call.";
    case Code.Unavailable:
      return "The server is not reachable through the demo.";
    case Code.DeadlineExceeded:
      return "The call took too long and was stopped.";
    case Code.InvalidArgument:
    case Code.FailedPrecondition:
    case Code.NotFound:
    case Code.AlreadyExists:
    case Code.OutOfRange:
    case Code.ResourceExhausted:
      return emptyField(message) || sentence(message) || `The server refused the call (${codeName(code)}).`;
    default:
      return sentence(message) || `The call failed (${codeName(code)}).`;
  }
}
function explain(e) {
  if (cancelled(e)) return { text: "Cancelled.", details: "" };
  if (!(e instanceof ConnectError)) {
    const m = e instanceof Error ? e.message : String(e);
    if (e instanceof TypeError) return { text: "The demo did not answer.", details: m };
    return { text: m, details: "" };
  }
  const message = unprefixed(e.rawMessage);
  const parts = [codeName(e.code)];
  const tc = e.metadata.get("turbo-code");
  const tf = e.metadata.get("turbo-field");
  if (tc) parts.push(`turbo-code ${tc}`);
  if (tf && tf !== "0") parts.push(`field ${tf}`);
  return { text: plain(e.code, message), details: `${parts.join(", ")}: ${e.rawMessage}` };
}

// demo/web/src/gen/open_inference_grpc_pb.ts
var file_open_inference_grpc = /* @__PURE__ */ fileDesc("ChlvcGVuX2luZmVyZW5jZV9ncnBjLnByb3RvEglpbmZlcmVuY2UiEwoRU2VydmVyTGl2ZVJlcXVlc3QiIgoSU2VydmVyTGl2ZVJlc3BvbnNlEgwKBGxpdmUYASABKAgiFAoSU2VydmVyUmVhZHlSZXF1ZXN0IiQKE1NlcnZlclJlYWR5UmVzcG9uc2USDQoFcmVhZHkYASABKAgiMgoRTW9kZWxSZWFkeVJlcXVlc3QSDAoEbmFtZRgBIAEoCRIPCgd2ZXJzaW9uGAIgASgJIiMKEk1vZGVsUmVhZHlSZXNwb25zZRINCgVyZWFkeRgBIAEoCCIXChVTZXJ2ZXJNZXRhZGF0YVJlcXVlc3QiSwoWU2VydmVyTWV0YWRhdGFSZXNwb25zZRIMCgRuYW1lGAEgASgJEg8KB3ZlcnNpb24YAiABKAkSEgoKZXh0ZW5zaW9ucxgDIAMoCSI1ChRNb2RlbE1ldGFkYXRhUmVxdWVzdBIMCgRuYW1lGAEgASgJEg8KB3ZlcnNpb24YAiABKAkihgMKFU1vZGVsTWV0YWRhdGFSZXNwb25zZRIMCgRuYW1lGAEgASgJEhAKCHZlcnNpb25zGAIgAygJEhAKCHBsYXRmb3JtGAMgASgJEj8KBmlucHV0cxgEIAMoCzIvLmluZmVyZW5jZS5Nb2RlbE1ldGFkYXRhUmVzcG9uc2UuVGVuc29yTWV0YWRhdGESQAoHb3V0cHV0cxgFIAMoCzIvLmluZmVyZW5jZS5Nb2RlbE1ldGFkYXRhUmVzcG9uc2UuVGVuc29yTWV0YWRhdGESRAoKcHJvcGVydGllcxgGIAMoCzIwLmluZmVyZW5jZS5Nb2RlbE1ldGFkYXRhUmVzcG9uc2UuUHJvcGVydGllc0VudHJ5Gj8KDlRlbnNvck1ldGFkYXRhEgwKBG5hbWUYASABKAkSEAoIZGF0YXR5cGUYAiABKAkSDQoFc2hhcGUYAyADKAMaMQoPUHJvcGVydGllc0VudHJ5EgsKA2tleRgBIAEoCRINCgV2YWx1ZRgCIAEoCToCOAEi7gYKEU1vZGVsSW5mZXJSZXF1ZXN0EhIKCm1vZGVsX25hbWUYASABKAkSFQoNbW9kZWxfdmVyc2lvbhgCIAEoCRIKCgJpZBgDIAEoCRJACgpwYXJhbWV0ZXJzGAQgAygLMiwuaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXF1ZXN0LlBhcmFtZXRlcnNFbnRyeRI9CgZpbnB1dHMYBSADKAsyLS5pbmZlcmVuY2UuTW9kZWxJbmZlclJlcXVlc3QuSW5mZXJJbnB1dFRlbnNvchJICgdvdXRwdXRzGAYgAygLMjcuaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXF1ZXN0LkluZmVyUmVxdWVzdGVkT3V0cHV0VGVuc29yEhoKEnJhd19pbnB1dF9jb250ZW50cxgHIAMoDBqUAgoQSW5mZXJJbnB1dFRlbnNvchIMCgRuYW1lGAEgASgJEhAKCGRhdGF0eXBlGAIgASgJEg0KBXNoYXBlGAMgAygDElEKCnBhcmFtZXRlcnMYBCADKAsyPS5pbmZlcmVuY2UuTW9kZWxJbmZlclJlcXVlc3QuSW5mZXJJbnB1dFRlbnNvci5QYXJhbWV0ZXJzRW50cnkSMAoIY29udGVudHMYBSABKAsyHi5pbmZlcmVuY2UuSW5mZXJUZW5zb3JDb250ZW50cxpMCg9QYXJhbWV0ZXJzRW50cnkSCwoDa2V5GAEgASgJEigKBXZhbHVlGAIgASgLMhkuaW5mZXJlbmNlLkluZmVyUGFyYW1ldGVyOgI4ARrVAQoaSW5mZXJSZXF1ZXN0ZWRPdXRwdXRUZW5zb3ISDAoEbmFtZRgBIAEoCRJbCgpwYXJhbWV0ZXJzGAIgAygLMkcuaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXF1ZXN0LkluZmVyUmVxdWVzdGVkT3V0cHV0VGVuc29yLlBhcmFtZXRlcnNFbnRyeRpMCg9QYXJhbWV0ZXJzRW50cnkSCwoDa2V5GAEgASgJEigKBXZhbHVlGAIgASgLMhkuaW5mZXJlbmNlLkluZmVyUGFyYW1ldGVyOgI4ARpMCg9QYXJhbWV0ZXJzRW50cnkSCwoDa2V5GAEgASgJEigKBXZhbHVlGAIgASgLMhkuaW5mZXJlbmNlLkluZmVyUGFyYW1ldGVyOgI4ASLVBAoSTW9kZWxJbmZlclJlc3BvbnNlEhIKCm1vZGVsX25hbWUYASABKAkSFQoNbW9kZWxfdmVyc2lvbhgCIAEoCRIKCgJpZBgDIAEoCRJBCgpwYXJhbWV0ZXJzGAQgAygLMi0uaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXNwb25zZS5QYXJhbWV0ZXJzRW50cnkSQAoHb3V0cHV0cxgFIAMoCzIvLmluZmVyZW5jZS5Nb2RlbEluZmVyUmVzcG9uc2UuSW5mZXJPdXRwdXRUZW5zb3ISGwoTcmF3X291dHB1dF9jb250ZW50cxgGIAMoDBqXAgoRSW5mZXJPdXRwdXRUZW5zb3ISDAoEbmFtZRgBIAEoCRIQCghkYXRhdHlwZRgCIAEoCRINCgVzaGFwZRgDIAMoAxJTCgpwYXJhbWV0ZXJzGAQgAygLMj8uaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXNwb25zZS5JbmZlck91dHB1dFRlbnNvci5QYXJhbWV0ZXJzRW50cnkSMAoIY29udGVudHMYBSABKAsyHi5pbmZlcmVuY2UuSW5mZXJUZW5zb3JDb250ZW50cxpMCg9QYXJhbWV0ZXJzRW50cnkSCwoDa2V5GAEgASgJEigKBXZhbHVlGAIgASgLMhkuaW5mZXJlbmNlLkluZmVyUGFyYW1ldGVyOgI4ARpMCg9QYXJhbWV0ZXJzRW50cnkSCwoDa2V5GAEgASgJEigKBXZhbHVlGAIgASgLMhkuaW5mZXJlbmNlLkluZmVyUGFyYW1ldGVyOgI4ASKZAQoOSW5mZXJQYXJhbWV0ZXISFAoKYm9vbF9wYXJhbRgBIAEoCEgAEhUKC2ludDY0X3BhcmFtGAIgASgDSAASFgoMc3RyaW5nX3BhcmFtGAMgASgJSAASFgoMZG91YmxlX3BhcmFtGAQgASgBSAASFgoMdWludDY0X3BhcmFtGAUgASgESABCEgoQcGFyYW1ldGVyX2Nob2ljZSLQAQoTSW5mZXJUZW5zb3JDb250ZW50cxIVCg1ib29sX2NvbnRlbnRzGAEgAygIEhQKDGludF9jb250ZW50cxgCIAMoBRIWCg5pbnQ2NF9jb250ZW50cxgDIAMoAxIVCg11aW50X2NvbnRlbnRzGAQgAygNEhcKD3VpbnQ2NF9jb250ZW50cxgFIAMoBBIVCg1mcDMyX2NvbnRlbnRzGAYgAygCEhUKDWZwNjRfY29udGVudHMYByADKAESFgoOYnl0ZXNfY29udGVudHMYCCADKAwy/AMKFEdSUENJbmZlcmVuY2VTZXJ2aWNlEksKClNlcnZlckxpdmUSHC5pbmZlcmVuY2UuU2VydmVyTGl2ZVJlcXVlc3QaHS5pbmZlcmVuY2UuU2VydmVyTGl2ZVJlc3BvbnNlIgASTgoLU2VydmVyUmVhZHkSHS5pbmZlcmVuY2UuU2VydmVyUmVhZHlSZXF1ZXN0Gh4uaW5mZXJlbmNlLlNlcnZlclJlYWR5UmVzcG9uc2UiABJLCgpNb2RlbFJlYWR5EhwuaW5mZXJlbmNlLk1vZGVsUmVhZHlSZXF1ZXN0Gh0uaW5mZXJlbmNlLk1vZGVsUmVhZHlSZXNwb25zZSIAElcKDlNlcnZlck1ldGFkYXRhEiAuaW5mZXJlbmNlLlNlcnZlck1ldGFkYXRhUmVxdWVzdBohLmluZmVyZW5jZS5TZXJ2ZXJNZXRhZGF0YVJlc3BvbnNlIgASVAoNTW9kZWxNZXRhZGF0YRIfLmluZmVyZW5jZS5Nb2RlbE1ldGFkYXRhUmVxdWVzdBogLmluZmVyZW5jZS5Nb2RlbE1ldGFkYXRhUmVzcG9uc2UiABJLCgpNb2RlbEluZmVyEhwuaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXF1ZXN0Gh0uaW5mZXJlbmNlLk1vZGVsSW5mZXJSZXNwb25zZSIAYgZwcm90bzM");
var InferParameterSchema = /* @__PURE__ */ messageDesc(file_open_inference_grpc, 12);
var GRPCInferenceService = /* @__PURE__ */ serviceDesc(file_open_inference_grpc, 0);

// demo/web/src/gen/turbo_pii_policy_pb.ts
var file_turbo_pii_policy = /* @__PURE__ */ fileDesc("ChZ0dXJib19waWlfcG9saWN5LnByb3RvEg90dXJiby5yZXN1bHQudjEi8wIKD1JlZGFjdGlvblBvbGljeRIKCgJpZBgBIAEoCRIQCghyZXZpc2lvbhgCIAEoBBIUCgxjdXN0b21fa2luZHMYAyADKAkSMAoKa2luZF9ydWxlcxgEIAMoCzIcLnR1cmJvLnJlc3VsdC52MS5QaWlLaW5kUnVsZRI9ChBkZWZhdWx0X3N0cmF0ZWd5GAUgASgLMiMudHVyYm8ucmVzdWx0LnYxLlBpaURlZmF1bHRTdHJhdGVneRIsCgVhbGxvdxgGIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5QaWlBbGxvd0xpc3QSKgoEZGVueRgHIAMoCzIcLnR1cmJvLnJlc3VsdC52MS5QaWlEZW55TGlzdBJAChJjdXN0b21faWRlbnRpZmllcnMYCCADKAsyJC50dXJiby5yZXN1bHQudjEuUGlpQ3VzdG9tSWRlbnRpZmllchIfChdyZWRhY3RfYW5kX2ZsYWdfdW5rbm93bhgJIAEoCCLMAgoLUGlpS2luZFJ1bGUSDwoHcnVsZV9pZBgBIAEoCRIpCgRraW5kGAIgASgLMhsudHVyYm8ucmVzdWx0LnYxLlBpaUtpbmRSZWYSLAoFc3RhdGUYAyABKA4yHS50dXJiby5yZXN1bHQudjEuUGlpUnVsZVN0YXRlEjUKCnN0cmF0ZWdpZXMYBCADKAsyIS50dXJiby5yZXN1bHQudjEuUGlpU3RyYXRlZ3lFbnRyeRIsCgVhbGxvdxgFIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5QaWlBbGxvd0xpc3QSMQoGbGFiZWxzGAYgASgLMiEudHVyYm8ucmVzdWx0LnYxLlBpaUxhYmVsT3ZlcnJpZGUSOwoPcmVsZWFzZV9jZWlsaW5nGAcgASgLMiIudHVyYm8ucmVzdWx0LnYxLlBpaVJlbGVhc2VDZWlsaW5nIkEKEVBpaVJlbGVhc2VDZWlsaW5nEiwKBWZvcm1zGAEgAygOMh0udHVyYm8ucmVzdWx0LnYxLlBpaVZhbHVlRm9ybSJ5ChBQaWlMYWJlbE92ZXJyaWRlEjkKC3JlcXVpcmVtZW50GAEgASgOMiQudHVyYm8ucmVzdWx0LnYxLlBpaUxhYmVsUmVxdWlyZW1lbnQSFAoMd2luZG93X3dvcmRzGAIgASgNEhQKDGV4dHJhX2xhYmVscxgDIAMoCSJ+ChBQaWlTdHJhdGVneUVudHJ5EjMKBHdoZW4YASADKAsyJS50dXJiby5yZXN1bHQudjEuUGlpU3RyYXRlZ3lDb25kaXRpb24SNQoIc3RyYXRlZ3kYAiABKAsyIy50dXJiby5yZXN1bHQudjEuUGlpUmV3cml0ZVN0cmF0ZWd5IqMCChRQaWlTdHJhdGVneUNvbmRpdGlvbhJGCg9jaGVja3N1bV9wYXNzZWQYASABKAsyKy50dXJiby5yZXN1bHQudjEuUGlpQ2hlY2tzdW1QYXNzZWRDb25kaXRpb25IABJCCg1sYWJlbF9wcmVzZW50GAIgASgLMikudHVyYm8ucmVzdWx0LnYxLlBpaUxhYmVsUHJlc2VudENvbmRpdGlvbkgAEjkKCGRldGVjdG9yGAMgASgLMiUudHVyYm8ucmVzdWx0LnYxLlBpaURldGVjdG9yQ29uZGl0aW9uSAASPAoOc2NvcmVfYXRfbGVhc3QYBCABKAsyIi50dXJiby5yZXN1bHQudjEuUGlpU2NvcmVDb25kaXRpb25IAEIGCgR0ZXN0IlcKGlBpaUNoZWNrc3VtUGFzc2VkQ29uZGl0aW9uEjkKCmFsZ29yaXRobXMYASADKA4yJS50dXJiby5yZXN1bHQudjEuUGlpQ2hlY2tzdW1BbGdvcml0aG0iGgoYUGlpTGFiZWxQcmVzZW50Q29uZGl0aW9uIjMKFFBpaURldGVjdG9yQ29uZGl0aW9uEgoKAmlkGAEgASgJEg8KB3ZlcnNpb24YAiABKAkiYQoRUGlpU2NvcmVDb25kaXRpb24SKwoEa2luZBgBIAEoDjIdLnR1cmJvLnJlc3VsdC52MS5QaWlTY29yZUtpbmQSCwoDbWluGAIgASgCEhIKCmNhbGlicmF0ZWQYAyABKAgi+gYKElBpaVJld3JpdGVTdHJhdGVneRIwCgRtYXNrGAEgASgLMiAudHVyYm8ucmVzdWx0LnYxLlBpaU1hc2tTdHJhdGVneUgAEjQKBmZvcm1hdBgCIAEoCzIiLnR1cmJvLnJlc3VsdC52MS5QaWlGb3JtYXRTdHJhdGVneUgAEi4KA21hcBgDIAEoCzIfLnR1cmJvLnJlc3VsdC52MS5QaWlNYXBTdHJhdGVneUgAEjgKCHRydW5jYXRlGAQgASgLMiQudHVyYm8ucmVzdWx0LnYxLlBpaVRydW5jYXRlU3RyYXRlZ3lIABIzCgZsYXN0X24YBSABKAsyIS50dXJiby5yZXN1bHQudjEuUGlpTGFzdE5TdHJhdGVneUgAEjwKCmFiYnJldmlhdGUYBiABKAsyJi50dXJiby5yZXN1bHQudjEuUGlpQWJicmV2aWF0ZVN0cmF0ZWd5SAASPQoLa2V5ZWRfdG9rZW4YByABKAsyJi50dXJiby5yZXN1bHQudjEuUGlpS2V5ZWRUb2tlblN0cmF0ZWd5SAASNAoGc2hhMjU2GAggASgLMiIudHVyYm8ucmVzdWx0LnYxLlBpaVNoYTI1NlN0cmF0ZWd5SAASOQoLc3RhdGljX3RleHQYCSABKAsyIi50dXJiby5yZXN1bHQudjEuUGlpU3RhdGljU3RyYXRlZ3lIABI6Cglwc2V1ZG9ueW0YCiABKAsyJS50dXJiby5yZXN1bHQudjEuUGlpUHNldWRvbnltU3RyYXRlZ3lIABJECg96aXBfc2FmZV9oYXJib3IYCyABKAsyKS50dXJiby5yZXN1bHQudjEuUGlpWmlwU2FmZUhhcmJvclN0cmF0ZWd5SAASRAoPYWdlX3NhZmVfaGFyYm9yGAwgASgLMikudHVyYm8ucmVzdWx0LnYxLlBpaUFnZVNhZmVIYXJib3JTdHJhdGVneUgAEjAKBGRhdGUYDSABKAsyIC50dXJiby5yZXN1bHQudjEuUGlpRGF0ZVN0cmF0ZWd5SAASOQoLYWVzXzI1Nl9nY20YDiABKAsyIi50dXJiby5yZXN1bHQudjEuUGlpQWVzR2NtU3RyYXRlZ3lIABIuCgNmZjEYDyABKAsyHy50dXJiby5yZXN1bHQudjEuUGlpRmYxU3RyYXRlZ3lIAEIKCghzdHJhdGVneSKRAQoSUGlpRGVmYXVsdFN0cmF0ZWd5EjAKBG1hc2sYASABKAsyIC50dXJiby5yZXN1bHQudjEuUGlpTWFza1N0cmF0ZWd5SAASPQoLa2V5ZWRfdG9rZW4YAiABKAsyJi50dXJiby5yZXN1bHQudjEuUGlpS2V5ZWRUb2tlblN0cmF0ZWd5SABCCgoIc3RyYXRlZ3kiOQoPUGlpTWFza1N0cmF0ZWd5EhEKCW1hc2tfY2hhchgBIAEoCRITCgtrZWVwX2Zvcm1hdBgCIAEoCCIlChFQaWlGb3JtYXRTdHJhdGVneRIQCgh0ZW1wbGF0ZRgBIAEoCSJ4ChNQaWlUcnVuY2F0ZVN0cmF0ZWd5EioKBHNpZGUYASABKA4yHC50dXJiby5yZXN1bHQudjEuUGlpS2VlcFNpZGUSDQoFY291bnQYAiABKA0SEQoJbWFza19jaGFyGAMgASgJEhMKC2tlZXBfZm9ybWF0GAQgASgIIkkKEFBpaUxhc3ROU3RyYXRlZ3kSDQoFY291bnQYASABKA0SEQoJbWFza19jaGFyGAIgASgJEhMKC2tlZXBfZm9ybWF0GAMgASgIIm0KFVBpaUFiYnJldmlhdGVTdHJhdGVneRIRCglzZXBhcmF0b3IYASABKAkSEAoIdHJhaWxpbmcYAiABKAkSGAoQcHJlc2VydmVfaHlwaGVucxgDIAEoCBIVCg1wcmVzZXJ2ZV9jYXNlGAQgASgIIkoKGFBpaVppcFNhZmVIYXJib3JTdHJhdGVneRIRCgltYXNrX2NoYXIYASABKAkSGwoTcmVzdHJpY3RlZF9wcmVmaXhlcxgCIAMoCSItChhQaWlBZ2VTYWZlSGFyYm9yU3RyYXRlZ3kSEQoJbWFza19jaGFyGAEgASgJIvsBCg9QaWlEYXRlU3RyYXRlZ3kSLAoEeWVhchgBIAEoCzIcLnR1cmJvLnJlc3VsdC52MS5QaWlEYXRlWWVhckgAEi4KBXNoaWZ0GAIgASgLMh0udHVyYm8ucmVzdWx0LnYxLlBpaURhdGVTaGlmdEgAEjQKCHJlbGF0aXZlGAMgASgLMiAudHVyYm8ucmVzdWx0LnYxLlBpaURhdGVSZWxhdGl2ZUgAEjkKC3NhZmVfaGFyYm9yGAUgASgLMiIudHVyYm8ucmVzdWx0LnYxLlBpaURhdGVTYWZlSGFyYm9ySAASEQoJbWFza19jaGFyGAQgASgJQgYKBG1vZGUiDQoLUGlpRGF0ZVllYXIijQEKDFBpaURhdGVTaGlmdBINCgV5ZWFycxgBIAEoBRIOCgZtb250aHMYAiABKAUSDAoEZGF5cxgDIAEoBRItCgNrZXkYBCABKAsyIC50dXJiby5yZXN1bHQudjEuUGlpUG9saWN5S2V5UmVmEhAKCG1heF9kYXlzGAUgASgNEg8KB2NvbnRleHQYBiABKAkiJAoPUGlpRGF0ZVJlbGF0aXZlEhEKCXJlZmVyZW5jZRgBIAEoCSImChFQaWlEYXRlU2FmZUhhcmJvchIRCglyZWZlcmVuY2UYASABKAkiUwoRUGlpQWVzR2NtU3RyYXRlZ3kSLQoDa2V5GAEgASgLMiAudHVyYm8ucmVzdWx0LnYxLlBpaVBvbGljeUtleVJlZhIPCgdjb250ZXh0GAIgASgJIoEBCg5QaWlGZjFTdHJhdGVneRItCgNrZXkYASABKAsyIC50dXJiby5yZXN1bHQudjEuUGlpUG9saWN5S2V5UmVmEjEKCGFscGhhYmV0GAIgASgOMh8udHVyYm8ucmVzdWx0LnYxLlBpaUZmMUFscGhhYmV0Eg0KBXR3ZWFrGAMgASgJIi4KD1BpaVBvbGljeUtleVJlZhIKCgJpZBgBIAEoCRIPCgd2ZXJzaW9uGAIgASgNIlYKFVBpaUtleWVkVG9rZW5TdHJhdGVneRItCgNrZXkYASABKAsyIC50dXJiby5yZXN1bHQudjEuUGlpUG9saWN5S2V5UmVmEg4KBmxlbmd0aBgCIAEoDSJIChFQaWlTaGEyNTZTdHJhdGVneRIOCgZsZW5ndGgYASABKA0SEwoLa2luZF9wcmVmaXgYAiABKAgSDgoGcHJlZml4GAMgASgJIigKEVBpaVN0YXRpY1N0cmF0ZWd5EhMKC3JlcGxhY2VtZW50GAEgASgJIjYKFFBpaVBzZXVkb255bVN0cmF0ZWd5Eg4KBnByZWZpeBgBIAEoCRIOCgZzdWZmaXgYAiABKAkigwIKDlBpaU1hcFN0cmF0ZWd5EjEKB2VudHJpZXMYASABKAsyHi50dXJiby5yZXN1bHQudjEuUGlpTWFwRW50cmllc0gAEjcKCmRpY3Rpb25hcnkYAiABKAsyIS50dXJiby5yZXN1bHQudjEuUGlpRGljdGlvbmFyeVJlZkgAEhYKDmNhc2Vfc2Vuc2l0aXZlGAMgASgIEi0KBnRhcmdldBgEIAEoDjIdLnR1cmJvLnJlc3VsdC52MS5QaWlNYXBUYXJnZXQSNQoIZmFsbGJhY2sYBSABKAsyIy50dXJiby5yZXN1bHQudjEuUGlpUmV3cml0ZVN0cmF0ZWd5QgcKBXRhYmxlIn0KDVBpaU1hcEVudHJpZXMSPAoHZW50cmllcxgBIAMoCzIrLnR1cmJvLnJlc3VsdC52MS5QaWlNYXBFbnRyaWVzLkVudHJpZXNFbnRyeRouCgxFbnRyaWVzRW50cnkSCwoDa2V5GAEgASgJEg0KBXZhbHVlGAIgASgJOgI4ASI3ChBQaWlEaWN0aW9uYXJ5UmVmEhMKC2FydGlmYWN0X2lkGAEgASgJEg4KBnNoYTI1NhgCIAEoCSJ+Cg1QaWlUZXJtU291cmNlEioKBXRlcm1zGAEgASgLMhkudHVyYm8ucmVzdWx0LnYxLlBpaVRlcm1zSAASNwoKZGljdGlvbmFyeRgCIAEoCzIhLnR1cmJvLnJlc3VsdC52MS5QaWlEaWN0aW9uYXJ5UmVmSABCCAoGc291cmNlIhkKCFBpaVRlcm1zEg0KBXRlcm1zGAEgAygJIm0KD1BpaUFsbG93UGF0dGVybhIxCgZzeW50YXgYASABKA4yIS50dXJiby5yZXN1bHQudjEuUGlpUGF0dGVyblN5bnRheBIPCgdwYXR0ZXJuGAIgASgJEhYKDmNhc2Vfc2Vuc2l0aXZlGAMgASgIIokBCgxQaWlBbGxvd0xpc3QSLQoFdGVybXMYASADKAsyHi50dXJiby5yZXN1bHQudjEuUGlpVGVybVNvdXJjZRIWCg5jYXNlX3NlbnNpdGl2ZRgCIAEoCBIyCghwYXR0ZXJucxgDIAMoCzIgLnR1cmJvLnJlc3VsdC52MS5QaWlBbGxvd1BhdHRlcm4ivQEKC1BpaURlbnlMaXN0Eg8KB3J1bGVfaWQYASABKAkSKQoEa2luZBgCIAEoCzIbLnR1cmJvLnJlc3VsdC52MS5QaWlLaW5kUmVmEi0KBXRlcm1zGAMgASgLMh4udHVyYm8ucmVzdWx0LnYxLlBpaVRlcm1Tb3VyY2USFgoOY2FzZV9zZW5zaXRpdmUYBCABKAgSEQoJbWF4X2VkaXRzGAUgASgNEhgKEGNhcGl0YWxpemVkX29ubHkYBiABKAgieAoTUGlpQ3VzdG9tSWRlbnRpZmllchIPCgdydWxlX2lkGAEgASgJEikKBGtpbmQYAiABKAsyGy50dXJiby5yZXN1bHQudjEuUGlpS2luZFJlZhINCgVzaGFwZRgDIAEoCRIWCg5jYXNlX3NlbnNpdGl2ZRgEIAEoCCJ2ChVQaWlSZWRhY3Rpb25TZWxlY3Rpb24SMgoGaW5saW5lGAEgASgLMiAudHVyYm8ucmVzdWx0LnYxLlJlZGFjdGlvblBvbGljeUgAEg4KBG5hbWUYAiABKAlIABIPCgdkcnlfcnVuGAMgASgIQggKBnBvbGljeSpnCgxQaWlSdWxlU3RhdGUSHgoaUElJX1JVTEVfU1RBVEVfVU5TUEVDSUZJRUQQABIaChZQSUlfUlVMRV9TVEFURV9FTkFCTEVEEAESGwoXUElJX1JVTEVfU1RBVEVfRElTQUJMRUQQAiqiAQoTUGlpTGFiZWxSZXF1aXJlbWVudBIlCiFQSUlfTEFCRUxfUkVRVUlSRU1FTlRfVU5TUEVDSUZJRUQQABIeChpQSUlfTEFCRUxfUkVRVUlSRU1FTlRfTk9ORRABEiIKHlBJSV9MQUJFTF9SRVFVSVJFTUVOVF9SRVFVSVJFRBACEiAKHFBJSV9MQUJFTF9SRVFVSVJFTUVOVF9CT09TVFMQAypcCgtQaWlLZWVwU2lkZRIdChlQSUlfS0VFUF9TSURFX1VOU1BFQ0lGSUVEEAASFwoTUElJX0tFRVBfU0lERV9TVEFSVBABEhUKEVBJSV9LRUVQX1NJREVfRU5EEAIqmwEKDlBpaUZmMUFscGhhYmV0EiAKHFBJSV9GRjFfQUxQSEFCRVRfVU5TUEVDSUZJRUQQABIbChdQSUlfRkYxX0FMUEhBQkVUX0RJR0lUUxABEicKI1BJSV9GRjFfQUxQSEFCRVRfVVBQRVJfQUxQSEFOVU1FUklDEAISIQodUElJX0ZGMV9BTFBIQUJFVF9BTFBIQU5VTUVSSUMQAyqTAQoMUGlpTWFwVGFyZ2V0Eh4KGlBJSV9NQVBfVEFSR0VUX1VOU1BFQ0lGSUVEEAASKAokUElJX01BUF9UQVJHRVRfTk9STUFMSVpFRF9PUl9TVVJGQUNFEAESHQoZUElJX01BUF9UQVJHRVRfTk9STUFMSVpFRBACEhoKFlBJSV9NQVBfVEFSR0VUX1NVUkZBQ0UQAypxChBQaWlQYXR0ZXJuU3ludGF4EiIKHlBJSV9QQVRURVJOX1NZTlRBWF9VTlNQRUNJRklFRBAAEhwKGFBJSV9QQVRURVJOX1NZTlRBWF9TSEFQRRABEhsKF1BJSV9QQVRURVJOX1NZTlRBWF9HTE9CEAJiBnByb3RvMw", [file_turbo_typed_annotations]);

// demo/web/src/gen/turbo_analysis_pb.ts
var file_turbo_analysis = /* @__PURE__ */ fileDesc("ChR0dXJib19hbmFseXNpcy5wcm90bxIPdHVyYm8ucmVzdWx0LnYxIhUKE0NhcGFiaWxpdGllc1JlcXVlc3Qi4gEKFENhcGFiaWxpdGllc1Jlc3BvbnNlEjQKB2J1bmRsZXMYASADKAsyIy50dXJiby5yZXN1bHQudjEuQnVuZGxlQ2FwYWJpbGl0aWVzEi0KBmxpbWl0cxgCIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5SZXN1bHRMaW1pdHMSLQoGc2VydmVyGAMgASgLMh0udHVyYm8ucmVzdWx0LnYxLlNlcnZlckxpbWl0cxI2Cghwcm9maWxlcxgEIAMoCzIkLnR1cmJvLnJlc3VsdC52MS5Qcm9maWxlQ2FwYWJpbGl0aWVzIm0KE1Byb2ZpbGVDYXBhYmlsaXRpZXMSDAoEbmFtZRgBIAEoCRIPCgd2ZXJzaW9uGAIgASgJEg0KBXN0ZXBzGAMgAygJEg4KBmxheWVycxgEIAMoCRIYChByZXF1aXJlZF9idW5kbGVzGAUgAygJIngKDFNlcnZlckxpbWl0cxIeChZtYXhfc2Vzc2lvbnNfcGVyX21vZGVsGAEgASgNEhkKEW1heF9sb2FkZWRfbW9kZWxzGAIgASgNEi0KBnN0cmVhbRgDIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5TdHJlYW1MaW1pdHMitAIKEkJ1bmRsZUNhcGFiaWxpdGllcxIMCgRuYW1lGAEgASgJEigKBW1vZGVsGAIgASgLMhkudHVyYm8ucmVzdWx0LnYxLk1vZGVsUmVmEgwKBHRhc2sYAyABKAkSDQoFcm9sZXMYBCADKAkSDgoGd2lkdGhzGAUgAygNEhUKDWNvbnRleHRfbGltaXQYBiABKA0SEwoLYmF0Y2hfbGltaXQYByABKA0SLQoGc2NoZW1hGAggASgLMh0udHVyYm8ucmVzdWx0LnYxLlNjaGVtYUxpbWl0cxISCgpoZWFkX3N0ZXBzGAkgAygJEjsKCnZvY2FidWxhcnkYCiABKAsyIi50dXJiby5yZXN1bHQudjEuVm9jYWJ1bGFyeUJpbmRpbmdIAIgBAUINCgtfdm9jYWJ1bGFyeSI8ChVWb2NhYnVsYXJ5QXJ0aWZhY3RSZWYSEwoLYXJ0aWZhY3RfaWQYASABKAkSDgoGc2hhMjU2GAIgASgJIq0BChFWb2NhYnVsYXJ5QmluZGluZxI+CglzdG9wd29yZHMYASABKAsyJi50dXJiby5yZXN1bHQudjEuVm9jYWJ1bGFyeUFydGlmYWN0UmVmSACIAQESPQoIZ2xvc3NhcnkYAiABKAsyJi50dXJiby5yZXN1bHQudjEuVm9jYWJ1bGFyeUFydGlmYWN0UmVmSAGIAQFCDAoKX3N0b3B3b3Jkc0ILCglfZ2xvc3NhcnkiogIKDFNjaGVtYUxpbWl0cxIYChBtYXhfZW50aXR5X3R5cGVzGAEgASgNEiAKGG1heF9jbGFzc2lmaWNhdGlvbl90YXNrcxgFIAEoDRIbChNtYXhfbGFiZWxzX3Blcl90YXNrGAYgASgNEhYKDm1heF9zdHJ1Y3R1cmVzGAcgASgNEiAKGG1heF9maWVsZHNfcGVyX3N0cnVjdHVyZRgIIAEoDRIWCg5tYXhfdGV4dF9ieXRlcxgJIAEoDRIbChNtYXhfcHJvbXB0X3N1YndvcmRzGAogASgNEhEKCW1heF93b3JkcxgLIAEoDUoECAIQA0oECAMQBEoECAQQBVIObWF4X2xhYmVsX3NldHNSCm1heF9maWVsZHNSCW1heF9kZXB0aCLaAgoMUmVzdWx0TGltaXRzEhoKEm1heF9zcGFyc2VfZW50cmllcxgBIAEoBBIeChZtYXhfc3BhcnNlX29jY3VycmVuY2VzGAIgASgEEh4KFm1heF90b2tlbl92ZWN0b3JfYnl0ZXMYAyABKAQSGAoQbWF4X3JlcmFua19pdGVtcxgEIAEoBBITCgttYXhfYW5zd2VycxgFIAEoBBIcChRtYXhfYW5zd2VyX2ZyYWdtZW50cxgGIAEoBBITCgttYXhfd2luZG93cxgHIAEoBBISCgptYXhfY2h1bmtzGAggASgEEhIKCm1heF9sZXZlbHMYCSABKAQSFQoNbWF4X2ZyYWdtZW50cxgKIAEoBBIXCg9tYXhfZ3JhcGhfYnl0ZXMYCyABKAQSGQoRbWF4X3JlY29yZF92YWx1ZXMYDCABKAQSGQoRbWF4X21hcHBpbmdfZWRpdHMYDSABKAQi+wEKEUxvYWRCdW5kbGVSZXF1ZXN0EgwKBG5hbWUYASABKAkSDgoGYnVuZGxlGAIgASgJEg4KBmRldmljZRgDIAEoDRIQCghzZXNzaW9ucxgEIAEoDRIRCglwcmVjaXNpb24YBSABKA0SEQoJbWF4X2JhdGNoGAYgASgNEg8KB21heF9zZXEYByABKA0SFQoNc2VsZWN0X2RldmljZRgLIAEoCBIMCgR1c2VyGAwgASgJEjsKCnZvY2FidWxhcnkYDSABKAsyIi50dXJiby5yZXN1bHQudjEuVm9jYWJ1bGFyeUJpbmRpbmdIAIgBAUINCgtfdm9jYWJ1bGFyeSJPChJMb2FkQnVuZGxlUmVzcG9uc2USOQoMY2FwYWJpbGl0aWVzGAEgASgLMiMudHVyYm8ucmVzdWx0LnYxLkJ1bmRsZUNhcGFiaWxpdGllcyIjChNVbmxvYWRCdW5kbGVSZXF1ZXN0EgwKBG5hbWUYASABKAkiFgoUVW5sb2FkQnVuZGxlUmVzcG9uc2UikAIKEVNwYXJzZVRlcm1zUmVzdWx0EigKBW1vZGVsGAEgASgLMhkudHVyYm8ucmVzdWx0LnYxLk1vZGVsUmVmEgsKA3JvdxgCIAMoDRIMCgR0ZXJtGAMgAygNEg4KBndlaWdodBgEIAMoAhIQCghoYXNfc3BhbhgFIAMoCBISCgpzcGFuX3N0YXJ0GAYgAygNEhAKCHNwYW5fZW5kGAcgAygNEhEKCW9jY19zdGFydBgIIAMoDRIPCgdvY2NfZW5kGAkgAygNEg4KBm9jY19hdBgKIAMoDRIMCgR0ZXh0GAsgASgJEiwKB29mZnNldHMYDCABKA4yGy50dXJiby5yZXN1bHQudjEuT2Zmc2V0TW9kZSLcAQoSVG9rZW5WZWN0b3JzUmVzdWx0EigKBW1vZGVsGAEgASgLMhkudHVyYm8ucmVzdWx0LnYxLk1vZGVsUmVmEgsKA2RpbRgCIAEoDRIMCgRkYXRhGAMgASgMEg0KBWVtcHR5GAQgAygNEgwKBHZpZXcYByABKA0SFAoMd2luZG93X3N0YXJ0GAUgAygNEhIKCndpbmRvd19lbmQYBiADKA0SDAoEdGV4dBgIIAEoCRIsCgdvZmZzZXRzGAkgASgOMhsudHVyYm8ucmVzdWx0LnYxLk9mZnNldE1vZGUihAEKElJlcmFua1Njb3Jlc1Jlc3VsdBIoCgVtb2RlbBgBIAEoCzIZLnR1cmJvLnJlc3VsdC52MS5Nb2RlbFJlZhIOCgZzb3VyY2UYAiABKAkSFwoPc291cmNlX2luc3RhbmNlGAUgASgJEgwKBGl0ZW0YAyADKA0SDQoFc2NvcmUYBCADKAIitQIKEEV4dHJhY3Rpb25SZXN1bHQSKAoFbW9kZWwYASABKAsyGS50dXJiby5yZXN1bHQudjEuTW9kZWxSZWYSDAoEdmlldxgCIAEoDRISCgpzcGFuX3N0YXJ0GAMgAygNEhAKCHNwYW5fZW5kGAQgAygNEg0KBXNjb3JlGAUgAygBEhMKC3Byb2JhYmlsaXR5GAYgAygBEhIKCm51bGxfc2NvcmUYByABKAESGAoQbnVsbF9wcm9iYWJpbGl0eRgIIAEoARISCgpmcmFnX3N0YXJ0GAkgAygNEhAKCGZyYWdfZW5kGAogAygNEg8KB2ZyYWdfYXQYCyADKA0SDAoEdGV4dBgMIAEoCRIsCgdvZmZzZXRzGA0gASgOMhsudHVyYm8ucmVzdWx0LnYxLk9mZnNldE1vZGUi5QQKDVJlY29yZHNSZXN1bHQSKAoFbW9kZWwYASABKAsyGS50dXJiby5yZXN1bHQudjEuTW9kZWxSZWYSDAoEdGV4dBgCIAEoCRIMCgR2aWV3GAMgASgNEiwKB29mZnNldHMYBCABKA4yGy50dXJiby5yZXN1bHQudjEuT2Zmc2V0TW9kZRISCgp0eXBlc19uYW1lGAUgASgJEg0KBXR5cGVzGAYgAygJEhIKCnBhdGhzX25hbWUYByABKAkSDQoFcGF0aHMYCCADKAkSFAoMY2hvaWNlc19uYW1lGAkgASgJEg8KB2Nob2ljZXMYCiADKAkSEwoLcmVjb3JkX3R5cGUYCyADKA0SFwoPcmVjb3JkX2luc3RhbmNlGAwgAygNEhgKEHZhbHVlX2ZyYWdfc3RhcnQYDSADKA0SFgoOdmFsdWVfZnJhZ19lbmQYDiADKA0SFQoNdmFsdWVfZnJhZ19hdBgPIAMoDRIUCgx2YWx1ZV9yZWNvcmQYECADKA0SEgoKdmFsdWVfcGF0aBgRIAMoDRITCgt2YWx1ZV9zY29yZRgSIAMoARITCgt2YWx1ZV9sb2dpdBgTIAMoARITCgtzdXBwb3J0X2VuZBgUIAMoDRIWCg5zdXBwb3J0X3NvdXJjZRgVIAMoDRITCgtzdXBwb3J0X3JvdxgWIAMoDRIVCg1jaG9pY2VfcmVjb3JkGBcgAygNEhMKC2Nob2ljZV9wYXRoGBggAygNEg4KBmNob2ljZRgZIAMoDRIUCgxjaG9pY2Vfc2NvcmUYGiADKAESFAoMY2hvaWNlX2xvZ2l0GBsgAygBIowBChRBbmFseXplU3RyZWFtUmVxdWVzdBI2CgZoZWFkZXIYASABKAsyJC50dXJiby5yZXN1bHQudjEuQW5hbHl6ZVN0cmVhbUhlYWRlckgAEjQKCGRvY3VtZW50GAIgASgLMiAudHVyYm8ucmVzdWx0LnYxLkFuYWx5emVEb2N1bWVudEgAQgYKBGtpbmQicgoTQW5hbHl6ZVN0cmVhbUhlYWRlchIsCgVvcmRlchgBIAEoDjIdLnR1cmJvLnJlc3VsdC52MS5BbmFseXplT3JkZXISFAoMbWF4X2luZmxpZ2h0GAIgASgNEhcKD2JhdGNoX2xpbmdlcl9tcxgDIAEoDSLwAQoOQW5hbHl6ZU9wdGlvbnMSEAoIdHJ1bmNhdGUYASABKA0SEgoKbWF4X3Rva2VucxgCIAEoDRITCgtwcm9tcHRfcm9sZRgDIAEoDRIRCglub3JtYWxpemUYBCABKA0SDwoHcG9vbGluZxgFIAEoDRISCgpvdXRwdXRfZGltGAYgASgNEhQKDHNwYXJzZV90b3BfaxgHIAEoDRIUCgxzcGFyc2VfZmxvb3IYCCABKAISFAoMdG9rZW5fc3RyaWRlGAkgASgNEhQKDHF1YW50aXphdGlvbhgKIAEoDRITCgtpbnQ4X3JhbmdlcxgLIAMoAiLXAgoPQW5hbHl6ZURvY3VtZW50EgoKAmlkGAEgASgJEg0KBW1vZGVsGAIgASgJEgwKBHRleHQYAyABKAwSMAoHb3B0aW9ucxgEIAEoCzIfLnR1cmJvLnJlc3VsdC52MS5BbmFseXplT3B0aW9ucxIuCgZvdXRwdXQYBSABKA4yHi50dXJiby5yZXN1bHQudjEuQW5hbHl6ZU91dHB1dBIsCgdvZmZzZXRzGAYgASgOMhsudHVyYm8ucmVzdWx0LnYxLk9mZnNldE1vZGUSMwoHYmluZGluZxgHIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5JbnB1dEJpbmRpbmdIAIgBARI7Cgp2b2NhYnVsYXJ5GAggASgLMiIudHVyYm8ucmVzdWx0LnYxLlZvY2FidWxhcnlCaW5kaW5nSAGIAQFCCgoIX2JpbmRpbmdCDQoLX3ZvY2FidWxhcnkizwMKFUFuYWx5emVTdHJlYW1SZXNwb25zZRIKCgJpZBgBIAEoCRIyCgd2ZWN0b3JzGAIgASgLMh8udHVyYm8ucmVzdWx0LnYxLkFuYWx5emVWZWN0b3JzSAASMgoHZmFpbHVyZRgDIAEoCzIfLnR1cmJvLnJlc3VsdC52MS5BbmFseXplRmFpbHVyZUgAEjoKDHNwYXJzZV90ZXJtcxgEIAEoCzIiLnR1cmJvLnJlc3VsdC52MS5TcGFyc2VUZXJtc1Jlc3VsdEgAEjwKDXRva2VuX3ZlY3RvcnMYBSABKAsyIy50dXJiby5yZXN1bHQudjEuVG9rZW5WZWN0b3JzUmVzdWx0SAASOgoMdmVjdG9yX2NvZGVzGAYgASgLMiIudHVyYm8ucmVzdWx0LnYxLlZlY3RvckNvZGVzUmVzdWx0SAASQAoOZG9jdW1lbnRfZ3JhcGgYCCABKAsyJi50dXJiby5yZXN1bHQudjEuRG9jdW1lbnRHcmFwaFJlc3BvbnNlSAASMwoHYmluZGluZxgHIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5JbnB1dEJpbmRpbmdIAYgBAUIJCgdvdXRjb21lQgoKCF9iaW5kaW5nImEKDkFuYWx5emVWZWN0b3JzEg0KBW1vZGVsGAEgASgJEg8KB3ZlcnNpb24YAiABKAkSCwoDZGltGAMgASgNEgwKBGRhdGEYBCABKAwSFAoMcXVhbnRpemF0aW9uGAUgASgNImQKEVZlY3RvckNvZGVzUmVzdWx0Eg0KBW1vZGVsGAEgASgJEg8KB3ZlcnNpb24YAiABKAkSCwoDZGltGAMgASgNEgwKBGRhdGEYBCABKAwSFAoMcXVhbnRpemF0aW9uGAUgASgNIkQKDkFuYWx5emVGYWlsdXJlEhIKCnR1cmJvX2NvZGUYASABKAUSDQoFZmllbGQYAiABKA0SDwoHbWVzc2FnZRgDIAEoCSL/AQoMU3RyZWFtTGltaXRzEiMKG21heF9xdWV1ZWRfaXRlbXNfcGVyX3N0cmVhbRgBIAEoBBIjChttYXhfcXVldWVkX2J5dGVzX3Blcl9zdHJlYW0YAiABKAQSHwoXbWF4X3F1ZXVlZF9pdGVtc19zZXJ2ZXIYAyABKAQSHwoXbWF4X3F1ZXVlZF9ieXRlc19zZXJ2ZXIYBCABKAQSJQodbWF4X2luZmxpZ2h0X2l0ZW1zX3Blcl9zdHJlYW0YBSABKAQSHwoXZGVmYXVsdF9iYXRjaF9saW5nZXJfbXMYBiABKAQSGwoTbWF4X2JhdGNoX2xpbmdlcl9tcxgHIAEoBCKgAgoJTGF5ZXJJbmZvEg0KBWxheWVyGAEgASgJEhAKCGluc3RhbmNlGAIgASgJEjAKBnN0YXR1cxgDIAEoDjIgLnR1cmJvLnJlc3VsdC52MS5MYXllcldpcmVTdGF0dXMSEQoJYW5ub3RhdG9yGAUgASgJEg8KB3ZlcnNpb24YBiABKAkSKAoFbW9kZWwYByABKAsyGS50dXJiby5yZXN1bHQudjEuTW9kZWxSZWYSEwoLY29uZmlnX2hhc2gYCCABKAQSDQoFaXRlbXMYCSABKA0SLwoFZXJyb3IYCiABKAsyGy50dXJiby5yZXN1bHQudjEuTGF5ZXJFcnJvckgAiAEBQggKBl9lcnJvckoECAQQBVINc3RhdHVzX2RldGFpbCJcChREb2N1bWVudEdyYXBoUmVxdWVzdBINCgVncmFwaBgBIAEoDBI1CgpwaWlfb3V0cHV0GAIgASgLMiEudHVyYm8ucmVzdWx0LnYxLlBpaU91dHB1dE9wdGlvbnMiyQEKFURvY3VtZW50R3JhcGhSZXNwb25zZRINCgVncmFwaBgBIAEoDBIqCgZsYXllcnMYAiADKAsyGi50dXJiby5yZXN1bHQudjEuTGF5ZXJJbmZvEigKBXZpZXdzGAMgAygLMhkudHVyYm8ucmVzdWx0LnYxLlZpZXdJbmZvEhMKC3NvdXJjZV90ZXh0GAQgASgJEjYKCGZpbmRpbmdzGAUgASgLMiQudHVyYm8ucmVzdWx0LnYxLlR5cGVkRG9jdW1lbnRSZXN1bHQisQYKEUNodW5rTGV2ZWxzUmVzdWx0EgwKBHRleHQYASABKAkSDAoEdmlldxgCIAEoDRIsCgdvZmZzZXRzGAMgASgOMhsudHVyYm8ucmVzdWx0LnYxLk9mZnNldE1vZGUSGAoQY2h1bmtfZnJhZ19zdGFydBgEIAMoDRIWCg5jaHVua19mcmFnX2VuZBgFIAMoDRIVCg1jaHVua19mcmFnX2F0GAYgAygNEhMKC2NodW5rX2xldmVsGAcgAygNEhQKDGNodW5rX3BhcmVudBgIIAMoBRIcChRjaHVua19zZW50ZW5jZV9zdGFydBgJIAMoDRIaChJjaHVua19zZW50ZW5jZV9lbmQYCiADKA0SEQoJY2h1bmtfY3V0GAsgAygNEhcKD2NodW5rX2N1dF9ibG9jaxgMIAMoDRISCgpjaHVua19jb3N0GA0gAygDEhoKEmNodW5rX21vZGVsX3Rva2VucxgYIAMoDRIZChFjaHVua19vdmVyX2J1ZGdldBgZIAMoCBIYChBsZXZlbF9mcmFnX3N0YXJ0GA4gAygNEhYKDmxldmVsX2ZyYWdfZW5kGA8gAygNEhUKDWxldmVsX2ZyYWdfYXQYECADKA0SEgoKbGV2ZWxfa2luZBgRIAMoDRIQCghsZXZlbF9peBgSIAMoDRIUCgxsZXZlbF9wYXJlbnQYEyADKAUSKAoFbW9kZWwYFCABKAsyGS50dXJiby5yZXN1bHQudjEuTW9kZWxSZWYSCwoDZGltGBUgASgNEgwKBGRhdGEYFiABKAwSDQoFZW1wdHkYFyADKA0SGAoQY2h1bmtfcGFnZV9zdGFydBgaIAMoDRIWCg5jaHVua19wYWdlX2VuZBgbIAMoDRIbChNjaHVua19oYXNfcGFnZV9zcGFuGBwgAygIEhUKDWNodW5rX2JveF9yb3cYHSADKA0SFgoOY2h1bmtfYm94X3BhZ2UYHiADKA0SFAoMY2h1bmtfYm94X3gwGB8gAygCEhQKDGNodW5rX2JveF95MBggIAMoAhIUCgxjaHVua19ib3hfeDEYISADKAISFAoMY2h1bmtfYm94X3kxGCIgAygCIpYFChZBbmFseXplUGlwZWxpbmVSZXF1ZXN0Eg4KBmRvY19pZBgBIAEoCRIMCgR0ZXh0GAIgASgJEhQKDHNvdXJjZV9ieXRlcxgDIAEoDBIsCgdvZmZzZXRzGAQgASgOMhsudHVyYm8ucmVzdWx0LnYxLk9mZnNldE1vZGUSDwoHcHJvZmlsZRgFIAEoCRINCgVzdGVwcxgGIAMoCRINCgVtb2RlbBgHIAEoCRIVCg1pbml0aWFsX2dyYXBoGAggASgMEjYKDnN0cnVjdHVyZV9zZWVkGAkgASgLMh4udHVyYm8ucmVzdWx0LnYxLlN0cnVjdHVyZVNlZWQSFQoNb3BlcmF0aW9uX2tleRgKIAEoCRI1CgpwaWlfb3V0cHV0GAsgASgLMiEudHVyYm8ucmVzdWx0LnYxLlBpaU91dHB1dE9wdGlvbnMSPQoNcGlpX3JlZGFjdGlvbhgMIAEoCzImLnR1cmJvLnJlc3VsdC52MS5QaWlSZWRhY3Rpb25TZWxlY3Rpb24SMwoHYmluZGluZxgNIAEoCzIdLnR1cmJvLnJlc3VsdC52MS5JbnB1dEJpbmRpbmdIAIgBARIyCglvcGVuX3NwYW4YDiABKAsyHy50dXJiby5yZXN1bHQudjEuT3BlblNwYW5TY2hlbWESNgoJcmVsYXRpb25zGA8gASgLMiMudHVyYm8ucmVzdWx0LnYxLk9wZW5SZWxhdGlvblNjaGVtYRIWCg5yZWxhdGlvbl9tb2RlbBgQIAEoCRI7Cgp2b2NhYnVsYXJ5GBEgASgLMiIudHVyYm8ucmVzdWx0LnYxLlZvY2FidWxhcnlCaW5kaW5nSAGIAQFCCgoIX2JpbmRpbmdCDQoLX3ZvY2FidWxhcnkiZwoOT3BlblNwYW5TY2hlbWESDgoGbGFiZWxzGAEgAygJEhYKCXRocmVzaG9sZBgCIAEoAkgAiAEBEg4KBm5lc3RlZBgDIAEoCBIPCgdwcm9tcHRzGAQgAygJQgwKCl90aHJlc2hvbGQihwEKEk9wZW5SZWxhdGlvblNjaGVtYRIyCgZsYWJlbHMYASADKAsyIi50dXJiby5yZXN1bHQudjEuT3BlblJlbGF0aW9uTGFiZWwSFgoJdGhyZXNob2xkGAIgASgCSACIAQESFwoPd2l0aGluX3NlbnRlbmNlGAMgASgIQgwKCl90aHJlc2hvbGQieQoRT3BlblJlbGF0aW9uTGFiZWwSDAoEbmFtZRgBIAEoCRIWCgl0aHJlc2hvbGQYAiABKAJIAIgBARIRCglzeW1tZXRyaWMYAyABKAgSDgoGZG9tYWluGAQgAygJEg0KBXJhbmdlGAUgAygJQgwKCl90aHJlc2hvbGQiTgoNU3RydWN0dXJlU2VlZBIMCgR0ZXh0GAEgASgJEi8KBmJsb2NrcxgCIAMoCzIfLnR1cmJvLnJlc3VsdC52MS5TdHJ1Y3R1cmVCbG9jayKOAQoOU3RydWN0dXJlQmxvY2sSMQoEa2luZBgBIAEoDjIjLnR1cmJvLnJlc3VsdC52MS5TdHJ1Y3R1cmVCbG9ja0tpbmQSDQoFc3RhcnQYAiABKA0SCwoDZW5kGAMgASgNEg0KBWxldmVsGAQgASgNEhMKBnBhcmVudBgFIAEoDUgAiAEBQgkKB19wYXJlbnQi1gIKF0FuYWx5emVQcm9ncmVzc2l2ZUV2ZW50EhAKCHNlcXVlbmNlGAEgASgEEjMKB3N0YXJ0ZWQYAiABKAsyIC50dXJiby5yZXN1bHQudjEuQW5hbHlzaXNTdGFydGVkSAASOwoMbGF5ZXJzX3JlYWR5GAMgASgLMiMudHVyYm8ucmVzdWx0LnYxLkFuYWx5c2lzTGF5ZXJCYXRjaEgAEjsKC3N0ZXBfZmFpbGVkGAQgASgLMiQudHVyYm8ucmVzdWx0LnYxLkFuYWx5c2lzU3RlcEZhaWx1cmVIABI6Cghjb21wbGV0ZRgFIAEoCzImLnR1cmJvLnJlc3VsdC52MS5Eb2N1bWVudEdyYXBoUmVzcG9uc2VIABI1CgdyZWNlaXB0GAYgASgLMiIudHVyYm8ucmVzdWx0LnYxLlByb2Nlc3NpbmdSZWNlaXB0SABCBwoFZXZlbnQiiwEKD0FuYWx5c2lzU3RhcnRlZBIOCgZkb2NfaWQYASABKAkSLAoHb2Zmc2V0cxgCIAEoDjIbLnR1cmJvLnJlc3VsdC52MS5PZmZzZXRNb2RlEhAKCHN0ZXBfaWRzGAMgAygJEigKBXZpZXdzGAQgAygLMhkudHVyYm8ucmVzdWx0LnYxLlZpZXdJbmZvIrUBChJBbmFseXNpc0xheWVyQmF0Y2gSFgoOY29tcGxldGVkX3N0ZXAYASABKAkSDQoFc3RhZ2UYAiABKA0SEwoLZHVyYXRpb25fdXMYAyABKAQSKgoGbGF5ZXJzGAQgAygLMhoudHVyYm8ucmVzdWx0LnYxLkxheWVySW5mbxIoCgV2aWV3cxgFIAMoCzIZLnR1cmJvLnJlc3VsdC52MS5WaWV3SW5mbxINCgVncmFwaBgGIAEoDCLwAQoTQW5hbHlzaXNTdGVwRmFpbHVyZRIPCgdzdGVwX2lkGAEgASgJEg0KBXN0YWdlGAIgASgNEhMKC2R1cmF0aW9uX3VzGAMgASgEEjAKB2ZhaWx1cmUYBCABKAsyHy50dXJiby5yZXN1bHQudjEuQW5hbHl6ZUZhaWx1cmUSGgoSc2tpcHBlZF9kZXBlbmRlbnRzGAUgAygJEioKBmxheWVycxgGIAMoCzIaLnR1cmJvLnJlc3VsdC52MS5MYXllckluZm8SKgoFZXJyb3IYByABKAsyGy50dXJiby5yZXN1bHQudjEuTGF5ZXJFcnJvciI2ChNMYXllckluc3RhbmNlRmlsdGVyEg0KBWxheWVyGAEgASgJEhAKCGluc3RhbmNlGAIgASgJIuQCChVEb2N1bWVudFdpbmRvd1JlcXVlc3QSDQoFZ3JhcGgYASABKAwSDAoEdmlldxgCIAEoDRIsCgdvZmZzZXRzGAMgASgOMhsudHVyYm8ucmVzdWx0LnYxLk9mZnNldE1vZGUSEQoJcGFnZV9zaXplGAQgASgNEgwKBHBhZ2UYBSABKA0SFQoNZmlsdGVyX2xheWVycxgGIAMoCRJEChZmaWx0ZXJfbGF5ZXJfaW5zdGFuY2VzGAogAygLMiQudHVyYm8ucmVzdWx0LnYxLkxheWVySW5zdGFuY2VGaWx0ZXISFwoPaGVhdG1hcF9idWNrZXRzGAcgASgNEh0KEHhyYXlfdGFyZ2V0X3ZpZXcYCCABKA1IAIgBARI1CgpwaWlfb3V0cHV0GAkgASgLMiEudHVyYm8ucmVzdWx0LnYxLlBpaU91dHB1dE9wdGlvbnNCEwoRX3hyYXlfdGFyZ2V0X3ZpZXci0gEKGURvY3VtZW50U2hhcGVTdW1tYXJ5UHJvdG8SDAoEdmlldxgBIAEoDRITCgt0b3RhbF9ieXRlcxgCIAEoDRITCgt0b3RhbF91bml0cxgDIAEoDRISCgp2aWV3X2NvdW50GAQgASgNEhMKC2xheWVyX2NvdW50GAUgASgNEioKBmxheWVycxgGIAMoCzIaLnR1cmJvLnJlc3VsdC52MS5MYXllckluZm8SKAoFdmlld3MYByADKAsyGS50dXJiby5yZXN1bHQudjEuVmlld0luZm8ipgEKE0RvY3VtZW50V2luZG93UHJvdG8SDAoEdmlldxgBIAEoDRISCgpieXRlX3N0YXJ0GAIgASgNEhAKCGJ5dGVfZW5kGAMgASgNEg0KBXN0YXJ0GAQgASgNEgsKA2VuZBgFIAEoDRIMCgRwYWdlGAYgASgNEhIKCnBhZ2VfY291bnQYByABKA0SDwoHY2xhbXBlZBgIIAEoCBIMCgR0ZXh0GAkgASgJImUKFlNlZ21lbnRBbm5vdGF0aW9uUHJvdG8SDQoFbGF5ZXIYASABKAkSEAoIaW5zdGFuY2UYAiABKAkSDAoEaXRlbRgDIAEoDRINCgVsYWJlbBgEIAEoCRINCgVzY29yZRgFIAEoASJ+ChRDb21iaW5lZFNlZ21lbnRQcm90bxINCgVzdGFydBgBIAEoDRILCgNlbmQYAiABKA0SDAoEdGV4dBgDIAEoCRI8Cgthbm5vdGF0aW9ucxgEIAMoCzInLnR1cmJvLnJlc3VsdC52MS5TZWdtZW50QW5ub3RhdGlvblByb3RvIlIKEURvY3VtZW50Q2hpcFByb3RvEg0KBWxheWVyGAEgASgJEhAKCGluc3RhbmNlGAIgASgJEg0KBWxhYmVsGAMgASgJEg0KBXNjb3JlGAQgASgBIqABChJIZWF0bWFwQnVja2V0UHJvdG8SDQoFc3RhcnQYASABKA0SCwoDZW5kGAIgASgNEg0KBXNjb3JlGAMgASgCEg0KBWNvdW50GAQgASgNEhkKDHByaW1hcnlfaXRlbRgFIAEoDUgAiAEBEhIKCmJ5dGVfc3RhcnQYBiABKA0SEAoIYnl0ZV9lbmQYByABKA1CDwoNX3ByaW1hcnlfaXRlbSKWAQoQSGVhdG1hcExhbmVQcm90bxIKCgJpZBgBIAEoCRIMCgRraW5kGAIgASgJEgwKBHZpZXcYAyABKA0SEQoJbWluX3Njb3JlGAQgASgCEhEKCW1heF9zY29yZRgFIAEoAhI0CgdidWNrZXRzGAYgAygLMiMudHVyYm8ucmVzdWx0LnYxLkhlYXRtYXBCdWNrZXRQcm90byK/AQoQWHJheUVkaXRSdW5Qcm90bxIUCgxwYXJlbnRfc3RhcnQYASABKA0SEgoKcGFyZW50X2VuZBgCIAEoDRITCgtjaGlsZF9zdGFydBgDIAEoDRIRCgljaGlsZF9lbmQYBCABKA0SMAoEa2luZBgFIAEoDjIiLnR1cmJvLnJlc3VsdC52MS5YcmF5RWRpdEtpbmRQcm90bxITCgtwYXJlbnRfdGV4dBgGIAEoCRISCgpjaGlsZF90ZXh0GAcgASgJItgBChRYcmF5U3RlcFN1bW1hcnlQcm90bxIMCgR2aWV3GAEgASgNEhgKC3BhcmVudF92aWV3GAIgASgNSACIAQESEAoIaW5zdGFuY2UYAyABKAkSEQoJYW5ub3RhdG9yGAQgASgJEg8KB3ZlcnNpb24YBSABKAkSEwoLY29uZmlnX2hhc2gYBiABKAQSEgoKZWRpdF9jb3VudBgHIAEoDRIUCgxwYXJlbnRfYnl0ZXMYCCABKA0SEwoLY2hpbGRfYnl0ZXMYCSABKA1CDgoMX3BhcmVudF92aWV3ItEBChZOb3JtYWxpemF0aW9uWHJheVByb3RvEhEKCWZyb21fdmlldxgBIAEoDRIPCgd0b192aWV3GAIgASgNEiwKB29mZnNldHMYAyABKA4yGy50dXJiby5yZXN1bHQudjEuT2Zmc2V0TW9kZRIvCgRydW5zGAQgAygLMiEudHVyYm8ucmVzdWx0LnYxLlhyYXlFZGl0UnVuUHJvdG8SNAoFc3RlcHMYBSADKAsyJS50dXJiby5yZXN1bHQudjEuWHJheVN0ZXBTdW1tYXJ5UHJvdG8i/wIKFkRvY3VtZW50V2luZG93UmVzcG9uc2USOwoHc3VtbWFyeRgBIAEoCzIqLnR1cmJvLnJlc3VsdC52MS5Eb2N1bWVudFNoYXBlU3VtbWFyeVByb3RvEjQKBndpbmRvdxgCIAEoCzIkLnR1cmJvLnJlc3VsdC52MS5Eb2N1bWVudFdpbmRvd1Byb3RvEjcKCHNlZ21lbnRzGAMgAygLMiUudHVyYm8ucmVzdWx0LnYxLkNvbWJpbmVkU2VnbWVudFByb3RvEjoKDmRvY3VtZW50X2NoaXBzGAQgAygLMiIudHVyYm8ucmVzdWx0LnYxLkRvY3VtZW50Q2hpcFByb3RvEjgKDWhlYXRtYXBfbGFuZXMYBSADKAsyIS50dXJiby5yZXN1bHQudjEuSGVhdG1hcExhbmVQcm90bxI6CgR4cmF5GAYgASgLMicudHVyYm8ucmVzdWx0LnYxLk5vcm1hbGl6YXRpb25YcmF5UHJvdG9IAIgBAUIHCgVfeHJheSJsCgpCdWRnZXRTcGVjEgwKBG5hbWUYASABKAkSEgoKbWF4X3Rva2VucxgCIAEoDRIVCg10YXJnZXRfdG9rZW5zGAMgASgNEhMKC3NpemVfd2VpZ2h0GAQgASgNEhAKCG92ZXJoZWFkGAUgASgNIq0CChRQcm9qZWN0Q2h1bmtzUmVxdWVzdBINCgVncmFwaBgBIAEoDBIsCgdvZmZzZXRzGAIgASgOMhsudHVyYm8ucmVzdWx0LnYxLk9mZnNldE1vZGUSEAoIcGNhX2RpbXMYAyABKA0SGgoSbWF4X3N0YWNrX3NlZ21lbnRzGAQgASgNEhsKE2luY2x1ZGVfZHJpZnRfY3VydmUYBSABKAgSHwoXaW5jbHVkZV9zdHJhdGVneV9tYXRyaXgYBiABKAgSNQoQc3RyYXRlZ3lfYnVkZ2V0cxgHIAMoCzIbLnR1cmJvLnJlc3VsdC52MS5CdWRnZXRTcGVjEjUKCnBpaV9vdXRwdXQYCCABKAsyIS50dXJiby5yZXN1bHQudjEuUGlpT3V0cHV0T3B0aW9ucyKwAQoTUHJvamVjdGVkUG9pbnRQcm90bxIMCgRpdGVtGAEgASgNEg0KBXN0YXJ0GAIgASgNEgsKA2VuZBgDIAEoDRINCgVsZXZlbBgEIAEoDRIJCgF4GAUgASgCEgkKAXkYBiABKAISCQoBehgHIAEoAhINCgVlbXB0eRgIIAEoCBIcCg9hZGphY2VudF9jb3NpbmUYCSABKAJIAIgBAUISChBfYWRqYWNlbnRfY29zaW5lIqUBChVWZWN0b3JQcm9qZWN0aW9uUHJvdG8SKAoFbW9kZWwYASABKAsyGS50dXJiby5yZXN1bHQudjEuTW9kZWxSZWYSEAoIcGNhX2RpbXMYAiABKA0SGgoSZXhwbGFpbmVkX3ZhcmlhbmNlGAMgAygCEjQKBnBvaW50cxgEIAMoCzIkLnR1cmJvLnJlc3VsdC52MS5Qcm9qZWN0ZWRQb2ludFByb3RvIvsBChJEcmlmdEdhcFBvaW50UHJvdG8SEAoIc2VudGVuY2UYASABKA0SEgoKZ2FwX29mZnNldBgCIAEoDRIdChByYXdfY29zaW5lX21pY3JvGAMgASgDSACIAQESIgoVc21vb3RoZWRfY29zaW5lX21pY3JvGAQgASgDSAGIAQESGQoRZGVwdGhfc2NvcmVfbWljcm8YBSABKAMSDAoEY29zdBgGIAEoBRIOCgZyZWFzb24YByABKA0SFAoMaXNfY2h1bmtfY3V0GAggASgIQhMKEV9yYXdfY29zaW5lX21pY3JvQhgKFl9zbW9vdGhlZF9jb3NpbmVfbWljcm8ibwoXQm91bmRhcnlEcmlmdEN1cnZlUHJvdG8SDgoGd2luZG93GAEgASgNEhEKCXNtb290aGluZxgCIAEoDRIxCgRnYXBzGAMgAygLMiMudHVyYm8ucmVzdWx0LnYxLkRyaWZ0R2FwUG9pbnRQcm90byKiAQoWVGVybVdlaWdodFNlZ21lbnRQcm90bxIMCgR0ZXJtGAEgASgJEg8KB3Rlcm1faWQYAiABKA0SEQoJZnJlcXVlbmN5GAMgASgNEg4KBndlaWdodBgEIAEoAhINCgVzaGFyZRgFIAEoAhITCgtpc19leHBhbmRlZBgGIAEoCBIRCglvY2Nfc3RhcnQYByADKA0SDwoHb2NjX2VuZBgIIAMoDSLsAQoUVGVybVZlY3RvclN0YWNrUHJvdG8SDAoEdW5pdBgBIAEoDRI5CghzZWdtZW50cxgCIAMoCzInLnR1cmJvLnJlc3VsdC52MS5UZXJtV2VpZ2h0U2VnbWVudFByb3RvEhMKC3RvdGFsX3Rlcm1zGAMgASgNEhcKD3RvdGFsX2ZyZXF1ZW5jeRgEIAEoBBIUCgx0b3RhbF93ZWlnaHQYBSABKAISGAoQb3RoZXJfdGVybV9jb3VudBgGIAEoDRIXCg9vdGhlcl9mcmVxdWVuY3kYByABKAQSFAoMb3RoZXJfd2VpZ2h0GAggASgCIvoBChxTdHJhdGVneUNvbXBhcmlzb25FbnRyeVByb3RvEgwKBG5hbWUYASABKAkSKwoGYnVkZ2V0GAIgASgLMhsudHVyYm8ucmVzdWx0LnYxLkJ1ZGdldFNwZWMSEwoLY2h1bmtfY291bnQYAyABKA0SGQoRb3Zlcl9idWRnZXRfY291bnQYBCABKA0SEgoKdG90YWxfY29zdBgFIAEoAxIiChptYXRjaGVkX2N1dHNfd2l0aF9iYXNlbGluZRgGIAEoDRIjChtkaXNhZ3JlZW1lbnRzX3dpdGhfYmFzZWxpbmUYByABKA0SEgoKY3V0X3N0YXJ0cxgIIAMoDSJYChNTdHJhdGVneU1hdHJpeFByb3RvEkEKCnN0cmF0ZWdpZXMYASADKAsyLS50dXJiby5yZXN1bHQudjEuU3RyYXRlZ3lDb21wYXJpc29uRW50cnlQcm90byKiAwoVUHJvamVjdENodW5rc1Jlc3BvbnNlEjoKCnByb2plY3Rpb24YASABKAsyJi50dXJiby5yZXN1bHQudjEuVmVjdG9yUHJvamVjdGlvblByb3RvEkIKC2RyaWZ0X2N1cnZlGAIgASgLMigudHVyYm8ucmVzdWx0LnYxLkJvdW5kYXJ5RHJpZnRDdXJ2ZVByb3RvSACIAQESQAoRY2h1bmtfdGVybV9zdGFja3MYAyADKAsyJS50dXJiby5yZXN1bHQudjEuVGVybVZlY3RvclN0YWNrUHJvdG8SRwoTZG9jdW1lbnRfdGVybV9zdGFjaxgEIAEoCzIlLnR1cmJvLnJlc3VsdC52MS5UZXJtVmVjdG9yU3RhY2tQcm90b0gBiAEBEkIKD3N0cmF0ZWd5X21hdHJpeBgFIAEoCzIkLnR1cmJvLnJlc3VsdC52MS5TdHJhdGVneU1hdHJpeFByb3RvSAKIAQFCDgoMX2RyaWZ0X2N1cnZlQhYKFF9kb2N1bWVudF90ZXJtX3N0YWNrQhIKEF9zdHJhdGVneV9tYXRyaXgiGgoYTGlzdE91dHB1dEZvcm1hdHNSZXF1ZXN0IsIBChtPdXRwdXRGb3JtYXREZXNjcmlwdG9yUHJvdG8SEQoJZm9ybWF0X2lkGAEgASgJEhQKDGRpc3BsYXlfbmFtZRgCIAEoCRISCgptZWRpYV90eXBlGAMgASgJEhYKDmZpbGVfZXh0ZW5zaW9uGAQgASgJEg4KBmJpbmFyeRgFIAEoCBIQCghsb3NzbGVzcxgGIAEoCBIXCg9yZXF1aXJlZF9sYXllcnMYByADKAkSEwoLZGVzY3JpcHRpb24YCCABKAkiWgoZTGlzdE91dHB1dEZvcm1hdHNSZXNwb25zZRI9Cgdmb3JtYXRzGAEgAygLMiwudHVyYm8ucmVzdWx0LnYxLk91dHB1dEZvcm1hdERlc2NyaXB0b3JQcm90byKuAQoVRm9ybWF0RG9jdW1lbnRSZXF1ZXN0Eg4KBmRvY19pZBgBIAEoCRINCgVncmFwaBgCIAEoDBIRCglmb3JtYXRfaWQYAyABKAkSLAoHb2Zmc2V0cxgEIAEoDjIbLnR1cmJvLnJlc3VsdC52MS5PZmZzZXRNb2RlEjUKCnBpaV9vdXRwdXQYBSABKAsyIS50dXJiby5yZXN1bHQudjEuUGlpT3V0cHV0T3B0aW9ucyJoChZGb3JtYXREb2N1bWVudFJlc3BvbnNlEhEKCWZvcm1hdF9pZBgBIAEoCRISCgptZWRpYV90eXBlGAIgASgJEhYKDmZpbGVfZXh0ZW5zaW9uGAMgASgJEg8KB2NvbnRlbnQYBCABKAwqRQoMQW5hbHl6ZU9yZGVyEhcKE0FOQUxZWkVfT1JERVJfSU5QVVQQABIcChhBTkFMWVpFX09SREVSX0NPTVBMRVRJT04QASqyAQoNQW5hbHl6ZU91dHB1dBIaChZBTkFMWVpFX09VVFBVVF9WRUNUT1JTEAASHwobQU5BTFlaRV9PVVRQVVRfU1BBUlNFX1RFUk1TEAESIAocQU5BTFlaRV9PVVRQVVRfVE9LRU5fVkVDVE9SUxACEh8KG0FOQUxZWkVfT1VUUFVUX1ZFQ1RPUl9DT0RFUxADEiEKHUFOQUxZWkVfT1VUUFVUX0RPQ1VNRU5UX0dSQVBIEAQq4gQKElN0cnVjdHVyZUJsb2NrS2luZBIkCiBTVFJVQ1RVUkVfQkxPQ0tfS0lORF9VTlNQRUNJRklFRBAAEiAKHFNUUlVDVFVSRV9CTE9DS19LSU5EX1NFQ1RJT04QARIgChxTVFJVQ1RVUkVfQkxPQ0tfS0lORF9IRUFESU5HEAISIgoeU1RSVUNUVVJFX0JMT0NLX0tJTkRfUEFSQUdSQVBIEAMSHQoZU1RSVUNUVVJFX0JMT0NLX0tJTkRfTElTVBAEEiIKHlNUUlVDVFVSRV9CTE9DS19LSU5EX0xJU1RfSVRFTRAFEh4KGlNUUlVDVFVSRV9CTE9DS19LSU5EX1RBQkxFEAYSIgoeU1RSVUNUVVJFX0JMT0NLX0tJTkRfVEFCTEVfUk9XEAcSIwofU1RSVUNUVVJFX0JMT0NLX0tJTkRfVEFCTEVfQ0VMTBAIEh0KGVNUUlVDVFVSRV9CTE9DS19LSU5EX0NPREUQCRIeChpTVFJVQ1RVUkVfQkxPQ0tfS0lORF9RVU9URRAKEiAKHFNUUlVDVFVSRV9CTE9DS19LSU5EX0NBUFRJT04QCxIhCh1TVFJVQ1RVUkVfQkxPQ0tfS0lORF9GT09UTk9URRAMEiQKIFNUUlVDVFVSRV9CTE9DS19LSU5EX1BBR0VfSEVBREVSEA0SJAogU1RSVUNUVVJFX0JMT0NLX0tJTkRfUEFHRV9GT09URVIQDhIjCh9TVFJVQ1RVUkVfQkxPQ0tfS0lORF9MT09TRV9MSVNUEA8SHQoZU1RSVUNUVVJFX0JMT0NLX0tJTkRfSFRNTBAQKpsBChFYcmF5RWRpdEtpbmRQcm90bxIeChpYUkFZX0VESVRfS0lORF9QUk9UT19FUVVBTBAAEiEKHVhSQVlfRURJVF9LSU5EX1BST1RPX1JFUExBQ0VEEAESIQodWFJBWV9FRElUX0tJTkRfUFJPVE9fSU5TRVJURUQQAhIgChxYUkFZX0VESVRfS0lORF9QUk9UT19ERUxFVEVEEAMyqA4KD0FuYWx5c2lzU2VydmljZRJdCgxDYXBhYmlsaXRpZXMSJC50dXJiby5yZXN1bHQudjEuQ2FwYWJpbGl0aWVzUmVxdWVzdBolLnR1cmJvLnJlc3VsdC52MS5DYXBhYmlsaXRpZXNSZXNwb25zZSIAElcKCkxvYWRCdW5kbGUSIi50dXJiby5yZXN1bHQudjEuTG9hZEJ1bmRsZVJlcXVlc3QaIy50dXJiby5yZXN1bHQudjEuTG9hZEJ1bmRsZVJlc3BvbnNlIgASXQoMVW5sb2FkQnVuZGxlEiQudHVyYm8ucmVzdWx0LnYxLlVubG9hZEJ1bmRsZVJlcXVlc3QaJS50dXJiby5yZXN1bHQudjEuVW5sb2FkQnVuZGxlUmVzcG9uc2UiABJgChRSb3VuZFRyaXBTcGFyc2VUZXJtcxIiLnR1cmJvLnJlc3VsdC52MS5TcGFyc2VUZXJtc1Jlc3VsdBoiLnR1cmJvLnJlc3VsdC52MS5TcGFyc2VUZXJtc1Jlc3VsdCIAEmMKFVJvdW5kVHJpcFRva2VuVmVjdG9ycxIjLnR1cmJvLnJlc3VsdC52MS5Ub2tlblZlY3RvcnNSZXN1bHQaIy50dXJiby5yZXN1bHQudjEuVG9rZW5WZWN0b3JzUmVzdWx0IgASYwoVUm91bmRUcmlwUmVyYW5rU2NvcmVzEiMudHVyYm8ucmVzdWx0LnYxLlJlcmFua1Njb3Jlc1Jlc3VsdBojLnR1cmJvLnJlc3VsdC52MS5SZXJhbmtTY29yZXNSZXN1bHQiABJdChNSb3VuZFRyaXBFeHRyYWN0aW9uEiEudHVyYm8ucmVzdWx0LnYxLkV4dHJhY3Rpb25SZXN1bHQaIS50dXJiby5yZXN1bHQudjEuRXh0cmFjdGlvblJlc3VsdCIAElQKEFJvdW5kVHJpcFJlY29yZHMSHi50dXJiby5yZXN1bHQudjEuUmVjb3Jkc1Jlc3VsdBoeLnR1cmJvLnJlc3VsdC52MS5SZWNvcmRzUmVzdWx0IgASYAoUUm91bmRUcmlwQ2h1bmtMZXZlbHMSIi50dXJiby5yZXN1bHQudjEuQ2h1bmtMZXZlbHNSZXN1bHQaIi50dXJiby5yZXN1bHQudjEuQ2h1bmtMZXZlbHNSZXN1bHQiABJpChZSb3VuZFRyaXBEb2N1bWVudEdyYXBoEiUudHVyYm8ucmVzdWx0LnYxLkRvY3VtZW50R3JhcGhSZXF1ZXN0GiYudHVyYm8ucmVzdWx0LnYxLkRvY3VtZW50R3JhcGhSZXNwb25zZSIAEmwKFEluc3BlY3REb2N1bWVudEdyYXBoEiwudHVyYm8ucmVzdWx0LnYxLkluc3BlY3REb2N1bWVudEdyYXBoUmVxdWVzdBokLnR1cmJvLnJlc3VsdC52MS5UeXBlZERvY3VtZW50UmVzdWx0IgASZgoWUm91bmRUcmlwVHlwZWREb2N1bWVudBIkLnR1cmJvLnJlc3VsdC52MS5UeXBlZERvY3VtZW50UmVzdWx0GiQudHVyYm8ucmVzdWx0LnYxLlR5cGVkRG9jdW1lbnRSZXN1bHQiABJkCg1BbmFseXplU3RyZWFtEiUudHVyYm8ucmVzdWx0LnYxLkFuYWx5emVTdHJlYW1SZXF1ZXN0GiYudHVyYm8ucmVzdWx0LnYxLkFuYWx5emVTdHJlYW1SZXNwb25zZSIAKAEwARJzChpBbmFseXplRG9jdW1lbnRQcm9ncmVzc2l2ZRInLnR1cmJvLnJlc3VsdC52MS5BbmFseXplUGlwZWxpbmVSZXF1ZXN0GigudHVyYm8ucmVzdWx0LnYxLkFuYWx5emVQcm9ncmVzc2l2ZUV2ZW50IgAwARJqChVJbnNwZWN0RG9jdW1lbnRXaW5kb3cSJi50dXJiby5yZXN1bHQudjEuRG9jdW1lbnRXaW5kb3dSZXF1ZXN0GicudHVyYm8ucmVzdWx0LnYxLkRvY3VtZW50V2luZG93UmVzcG9uc2UiABJgCg1Qcm9qZWN0Q2h1bmtzEiUudHVyYm8ucmVzdWx0LnYxLlByb2plY3RDaHVua3NSZXF1ZXN0GiYudHVyYm8ucmVzdWx0LnYxLlByb2plY3RDaHVua3NSZXNwb25zZSIAEmwKEUxpc3RPdXRwdXRGb3JtYXRzEikudHVyYm8ucmVzdWx0LnYxLkxpc3RPdXRwdXRGb3JtYXRzUmVxdWVzdBoqLnR1cmJvLnJlc3VsdC52MS5MaXN0T3V0cHV0Rm9ybWF0c1Jlc3BvbnNlIgASYwoORm9ybWF0RG9jdW1lbnQSJi50dXJiby5yZXN1bHQudjEuRm9ybWF0RG9jdW1lbnRSZXF1ZXN0GicudHVyYm8ucmVzdWx0LnYxLkZvcm1hdERvY3VtZW50UmVzcG9uc2UiAFAAYgZwcm90bzM", [file_turbo_typed_annotations, file_turbo_pii_policy]);
var AnalysisService = /* @__PURE__ */ serviceDesc(file_turbo_analysis, 0);

// demo/web/src/gen/turbo_demo_pb.ts
var file_turbo_demo = /* @__PURE__ */ fileDesc("ChB0dXJib19kZW1vLnByb3RvEg10dXJiby5kZW1vLnYxIhEKD0Rlc2NyaWJlUmVxdWVzdCI+ChBEZXNjcmliZVJlc3BvbnNlEhgKEGZvcndhcmRlZF93cml0ZXMYASADKAkSEAoIaGFzX3VzZXIYAiABKAgyWgoLRGVtb1NlcnZpY2USSwoIRGVzY3JpYmUSHi50dXJiby5kZW1vLnYxLkRlc2NyaWJlUmVxdWVzdBofLnR1cmJvLmRlbW8udjEuRGVzY3JpYmVSZXNwb25zZWIGcHJvdG8z");
var DemoService = /* @__PURE__ */ serviceDesc(file_turbo_demo, 0);

// demo/web/src/gen/turbo_search_pb.ts
var file_turbo_search = /* @__PURE__ */ fileDesc("ChJ0dXJib19zZWFyY2gucHJvdG8SD3R1cmJvLnNlYXJjaC52MSI9Cg1Nb2RlbElkZW50aXR5EgoKAmlkGAEgASgJEhAKCHJldmlzaW9uGAIgASgJEg4KBnNoYTI1NhgDIAEoCSKCAQoJU3BhY2VTcGVjEi0KBW1vZGVsGAEgASgLMh4udHVyYm8uc2VhcmNoLnYxLk1vZGVsSWRlbnRpdHkSCwoDZGltGAIgASgNEjkKDW5vcm1hbGl6YXRpb24YAyABKA4yIi50dXJiby5zZWFyY2gudjEuTm9ybWFsaXphdGlvbk1vZGUiqQIKCUluZGV4U3BlYxISCgppbmRleF9uYW1lGAEgASgJEikKBXNwYWNlGAIgASgLMhoudHVyYm8uc2VhcmNoLnYxLlNwYWNlU3BlYxIvCgZtZXRyaWMYAyABKA4yHy50dXJiby5zZWFyY2gudjEuRGlzdGFuY2VNZXRyaWMSMwoHc3RvcmFnZRgEIAEoDjIiLnR1cmJvLnNlYXJjaC52MS5WZWN0b3JTdG9yYWdlS2luZBISCgpxdWFudF9iaXRzGAUgASgNEhcKD3F1YW50X2NhbGlicmF0ZRgGIAEoCBIXCg9jb2xsZWN0aW9uX25hbWUYByABKAkSFAoMZW1iZWRfYnVuZGxlGAggASgJEhsKE2NodW5rX2J1ZGdldF90b2tlbnMYCSABKA0i7wIKDEluZGV4U3VtbWFyeRISCgppbmRleF9uYW1lGAEgASgJEikKBXNwYWNlGAIgASgLMhoudHVyYm8uc2VhcmNoLnYxLlNwYWNlU3BlYxIvCgZtZXRyaWMYAyABKA4yHy50dXJiby5zZWFyY2gudjEuRGlzdGFuY2VNZXRyaWMSMwoHc3RvcmFnZRgEIAEoDjIiLnR1cmJvLnNlYXJjaC52MS5WZWN0b3JTdG9yYWdlS2luZBISCgpxdWFudF9iaXRzGAUgASgNEg4KBnNlYWxlZBgGIAEoCBIWCg5kb2N1bWVudF9jb3VudBgHIAEoDRITCgtjaHVua19jb3VudBgIIAEoDRIXCg9kZW5zZV9yb3dfY291bnQYCSABKA0SGAoQc3BhcnNlX3Jvd19jb3VudBgKIAEoDRIdChV0b2tlbl9jYW5kaWRhdGVfY291bnQYCyABKA0SFwoPY29sbGVjdGlvbl9uYW1lGAwgASgJIkkKGFNlYXJjaFByb3ZpZGVyRGVzY3JpcHRvchIKCgJpZBgBIAEoCRIMCgRraW5kGAIgASgJEhMKC2Rlc2NyaXB0aW9uGAMgASgJIhwKGkxpc3RTZWFyY2hQcm92aWRlcnNSZXF1ZXN0IlsKG0xpc3RTZWFyY2hQcm92aWRlcnNSZXNwb25zZRI8Cglwcm92aWRlcnMYASADKAsyKS50dXJiby5zZWFyY2gudjEuU2VhcmNoUHJvdmlkZXJEZXNjcmlwdG9yIhQKEkxpc3RJbmRleGVzUmVxdWVzdCJFChNMaXN0SW5kZXhlc1Jlc3BvbnNlEi4KB2luZGV4ZXMYASADKAsyHS50dXJiby5zZWFyY2gudjEuSW5kZXhTdW1tYXJ5IlYKEkNyZWF0ZUluZGV4UmVxdWVzdBIoCgRzcGVjGAEgASgLMhoudHVyYm8uc2VhcmNoLnYxLkluZGV4U3BlYxIWCg5sb2FkX2Zyb21fcGF0aBgCIAEoCSJFChNDcmVhdGVJbmRleFJlc3BvbnNlEi4KB3N1bW1hcnkYASABKAsyHS50dXJiby5zZWFyY2gudjEuSW5kZXhTdW1tYXJ5IigKEkRlbGV0ZUluZGV4UmVxdWVzdBISCgppbmRleF9uYW1lGAEgASgJIhUKE0RlbGV0ZUluZGV4UmVzcG9uc2UifgoQU3BhcnNlRW50cnlQcm90bxIPCgd0ZXJtX2lkGAEgASgNEhEKCXRlcm1fdGV4dBgCIAEoCRIOCgZ3ZWlnaHQYAyABKAISEgoKc3Bhbl9zdGFydBgEIAEoDRIQCghzcGFuX2VuZBgFIAEoDRIQCghoYXNfc3BhbhgGIAEoCCJaChBUb2tlblZlY3RvcklucHV0Eg4KBnZhbHVlcxgBIAMoAhISCgpzcGFuX3N0YXJ0GAIgASgNEhAKCHNwYW5fZW5kGAMgASgNEhAKCGhhc19zcGFuGAQgASgIIlQKFFBhZ2VCb3VuZGluZ0JveFByb3RvEgwKBHBhZ2UYASABKA0SCgoCeDAYAiABKAISCgoCeTAYAyABKAISCgoCeDEYBCABKAISCgoCeTEYBSABKAIieAoOTGF5b3V0Qm94SW5wdXQSKAoEc3BhbhgBIAEoCzIaLnR1cmJvLnNlYXJjaC52MS5TcGFuUHJvdG8SDAoEcGFnZRgCIAEoDRIKCgJ4MBgDIAEoAhIKCgJ5MBgEIAEoAhIKCgJ4MRgFIAEoAhIKCgJ5MRgGIAEoAiKLBAoKQ2h1bmtJbnB1dBIQCghjaHVua19pZBgBIAEoCRISCgpsZXZlbF9raW5kGAIgASgJEhIKCnNwYW5fc3RhcnQYAyABKA0SEAoIc3Bhbl9lbmQYBCABKA0SFAoMcGFyZW50X3N0YXJ0GAUgASgNEhIKCnBhcmVudF9lbmQYBiABKA0SEgoKaGFzX3BhcmVudBgHIAEoCBIUCgxkZW5zZV92ZWN0b3IYCCADKAISNwoMc3BhcnNlX3Rlcm1zGAkgAygLMiEudHVyYm8uc2VhcmNoLnYxLlNwYXJzZUVudHJ5UHJvdG8SOAoNdG9rZW5fdmVjdG9ycxgKIAMoCzIhLnR1cmJvLnNlYXJjaC52MS5Ub2tlblZlY3RvcklucHV0Ej8KCmF0dHJpYnV0ZXMYCyADKAsyKy50dXJiby5zZWFyY2gudjEuQ2h1bmtJbnB1dC5BdHRyaWJ1dGVzRW50cnkSEgoKcGFnZV9zdGFydBgMIAEoDRIQCghwYWdlX2VuZBgNIAEoDRIVCg1oYXNfcGFnZV9zcGFuGA4gASgIEjkKCnBhZ2VfYm94ZXMYDyADKAsyJS50dXJiby5zZWFyY2gudjEuUGFnZUJvdW5kaW5nQm94UHJvdG8aMQoPQXR0cmlidXRlc0VudHJ5EgsKA2tleRgBIAEoCRINCgV2YWx1ZRgCIAEoCToCOAEiuQIKFEluZGV4RG9jdW1lbnRSZXF1ZXN0EhIKCmluZGV4X25hbWUYASABKAkSDgoGZG9jX2lkGAIgASgJEgwKBHRleHQYAyABKAkSDQoFZ3JhcGgYBCABKAwSKwoGY2h1bmtzGAUgAygLMhsudHVyYm8uc2VhcmNoLnYxLkNodW5rSW5wdXQSSQoKYXR0cmlidXRlcxgGIAMoCzI1LnR1cmJvLnNlYXJjaC52MS5JbmRleERvY3VtZW50UmVxdWVzdC5BdHRyaWJ1dGVzRW50cnkSNQoMbGF5b3V0X2JveGVzGAcgAygLMh8udHVyYm8uc2VhcmNoLnYxLkxheW91dEJveElucHV0GjEKD0F0dHJpYnV0ZXNFbnRyeRILCgNrZXkYASABKAkSDQoFdmFsdWUYAiABKAk6AjgBIpYBChZJbmRleERvY3VtZW50c1Jlc3BvbnNlEhIKCmluZGV4X25hbWUYASABKAkSGQoRaW5kZXhlZF9kb2N1bWVudHMYAiABKA0SFgoOaW5kZXhlZF9jaHVua3MYAyABKA0SNQoFZHJpZnQYBCABKAsyJi50dXJiby5zZWFyY2gudjEuVm9jYWJ1bGFyeURyaWZ0UmVwb3J0InoKDlNlbWFudGljQ2xhdXNlEhQKDHF1ZXJ5X3ZlY3RvchgBIAMoAhIvCgtxdWVyeV9zcGFjZRgCIAEoCzIaLnR1cmJvLnNlYXJjaC52MS5TcGFjZVNwZWMSEgoKcXVlcnlfdGV4dBgDIAEoCRINCgV0b3BfaxgEIAEoDSJYCgpUZXJtQ2xhdXNlEg0KBXRlcm1zGAEgAygJEiwKBG1vZGUYAiABKA4yHi50dXJiby5zZWFyY2gudjEuVGVybU1hdGNoTW9kZRINCgV0b3BfaxgDIAEoDSI6CgxQaHJhc2VDbGF1c2USDQoFdGVybXMYASADKAkSDAoEc2xvcBgCIAEoDRINCgV0b3BfaxgDIAEoDSJVCgxTcGFyc2VDbGF1c2USNgoLcXVlcnlfdGVybXMYASADKAsyIS50dXJiby5zZWFyY2gudjEuU3BhcnNlRW50cnlQcm90bxINCgV0b3BfaxgCIAEoDSJ0ChVMYXRlSW50ZXJhY3Rpb25DbGF1c2USGwoTcXVlcnlfdG9rZW5fdmVjdG9ycxgBIAMoAhIvCgtxdWVyeV9zcGFjZRgCIAEoCzIaLnR1cmJvLnNlYXJjaC52MS5TcGFjZVNwZWMSDQoFdG9wX2sYAyABKA0ivwEKDEZpbHRlckNsYXVzZRIOCgZkb2NfaWQYASABKAkSEgoKbGV2ZWxfa2luZBgCIAEoCRJBCgphdHRyaWJ1dGVzGAMgAygLMi0udHVyYm8uc2VhcmNoLnYxLkZpbHRlckNsYXVzZS5BdHRyaWJ1dGVzRW50cnkSFQoNY29udGFpbnNfdGV4dBgEIAEoCRoxCg9BdHRyaWJ1dGVzRW50cnkSCwoDa2V5GAEgASgJEg0KBXZhbHVlGAIgASgJOgI4ASJHCgtCb29zdENsYXVzZRIpCgVjaGlsZBgBIAEoCzIaLnR1cmJvLnNlYXJjaC52MS5RdWVyeU5vZGUSDQoFYm9vc3QYAiABKAIipQEKCkpvaW5DbGF1c2USKQoCb3AYASABKA4yHS50dXJiby5zZWFyY2gudjEuSm9pbk9wZXJhdG9yEi8KBmZ1c2lvbhgCIAEoDjIfLnR1cmJvLnNlYXJjaC52MS5GdXNpb25TdHJhdGVneRIsCghjaGlsZHJlbhgDIAMoCzIaLnR1cmJvLnNlYXJjaC52MS5RdWVyeU5vZGUSDQoFcnJmX2sYBCABKAIiqgMKCVF1ZXJ5Tm9kZRIzCghzZW1hbnRpYxgBIAEoCzIfLnR1cmJvLnNlYXJjaC52MS5TZW1hbnRpY0NsYXVzZUgAEisKBHRlcm0YAiABKAsyGy50dXJiby5zZWFyY2gudjEuVGVybUNsYXVzZUgAEi8KBnBocmFzZRgDIAEoCzIdLnR1cmJvLnNlYXJjaC52MS5QaHJhc2VDbGF1c2VIABIvCgZzcGFyc2UYBCABKAsyHS50dXJiby5zZWFyY2gudjEuU3BhcnNlQ2xhdXNlSAASQgoQbGF0ZV9pbnRlcmFjdGlvbhgFIAEoCzImLnR1cmJvLnNlYXJjaC52MS5MYXRlSW50ZXJhY3Rpb25DbGF1c2VIABIvCgZmaWx0ZXIYBiABKAsyHS50dXJiby5zZWFyY2gudjEuRmlsdGVyQ2xhdXNlSAASLQoFYm9vc3QYByABKAsyHC50dXJiby5zZWFyY2gudjEuQm9vc3RDbGF1c2VIABIrCgRqb2luGAggASgLMhsudHVyYm8uc2VhcmNoLnYxLkpvaW5DbGF1c2VIAEIICgZjbGF1c2UiQQoTUGFyZW50RXhwYW5zaW9uU3BlYxIPCgdlbmFibGVkGAEgASgIEhkKEW1heF9wYXJlbnRfdG9rZW5zGAIgASgNIpYCCg1TZWFyY2hSZXF1ZXN0EhIKCmluZGV4X25hbWUYASABKAkSFQoNbmF0dXJhbF9xdWVyeRgCIAEoCRItCglxdWVyeV9hc3QYAyABKAsyGi50dXJiby5zZWFyY2gudjEuUXVlcnlOb2RlEhQKDHF1ZXJ5X3ZlY3RvchgEIAMoAhIvCgtxdWVyeV9zcGFjZRgFIAEoCzIaLnR1cmJvLnNlYXJjaC52MS5TcGFjZVNwZWMSDQoFdG9wX2sYBiABKA0SPgoQcGFyZW50X2V4cGFuc2lvbhgHIAEoCzIkLnR1cmJvLnNlYXJjaC52MS5QYXJlbnRFeHBhbnNpb25TcGVjEhUKDXJlcmFua19idW5kbGUYCCABKAkiJwoJU3BhblByb3RvEg0KBXN0YXJ0GAEgASgNEgsKA2VuZBgCIAEoDSJlCgtNYXRjaGVkU3BhbhIOCgZzb3VyY2UYASABKAkSDQoFbGFiZWwYAiABKAkSKAoEc3BhbhgDIAEoCzIaLnR1cmJvLnNlYXJjaC52MS5TcGFuUHJvdG8SDQoFc2NvcmUYBCABKAIiyAEKEENodW5rQXR0cmlidXRpb24SMwoNbWF0Y2hlZF9zcGFucxgBIAMoCzIcLnR1cmJvLnNlYXJjaC52MS5NYXRjaGVkU3BhbhJKCg1zaWduYWxfc2NvcmVzGAIgAygLMjMudHVyYm8uc2VhcmNoLnYxLkNodW5rQXR0cmlidXRpb24uU2lnbmFsU2NvcmVzRW50cnkaMwoRU2lnbmFsU2NvcmVzRW50cnkSCwoDa2V5GAEgASgJEg0KBXZhbHVlGAIgASgCOgI4ASKLBAoJU2VhcmNoSGl0EhAKCGNodW5rX2lkGAEgASgJEg4KBmRvY19pZBgCIAEoCRISCgpsZXZlbF9raW5kGAMgASgJEg0KBXNjb3JlGAQgASgCEi4KCmNodW5rX3NwYW4YBSABKAsyGi50dXJiby5zZWFyY2gudjEuU3BhblByb3RvEi8KC3BhcmVudF9zcGFuGAYgASgLMhoudHVyYm8uc2VhcmNoLnYxLlNwYW5Qcm90bxISCgpoYXNfcGFyZW50GAcgASgIEgwKBHRleHQYCCABKAkSEwoLcGFyZW50X3RleHQYCSABKAkSNgoLYXR0cmlidXRpb24YCiABKAsyIS50dXJiby5zZWFyY2gudjEuQ2h1bmtBdHRyaWJ1dGlvbhI+CgphdHRyaWJ1dGVzGAsgAygLMioudHVyYm8uc2VhcmNoLnYxLlNlYXJjaEhpdC5BdHRyaWJ1dGVzRW50cnkSEgoKcGFnZV9zdGFydBgMIAEoDRIQCghwYWdlX2VuZBgNIAEoDRIVCg1oYXNfcGFnZV9zcGFuGA4gASgIEjkKCnBhZ2VfYm94ZXMYDyADKAsyJS50dXJiby5zZWFyY2gudjEuUGFnZUJvdW5kaW5nQm94UHJvdG8aMQoPQXR0cmlidXRlc0VudHJ5EgsKA2tleRgBIAEoCRINCgV2YWx1ZRgCIAEoCToCOAEiWAoLSGVhdFNlZ21lbnQSKAoEc3BhbhgBIAEoCzIaLnR1cmJvLnNlYXJjaC52MS5TcGFuUHJvdG8SDQoFc2NvcmUYAiABKAISEAoIY2h1bmtfaWQYAyABKAkiYQoMRG9jdW1lbnRIZWF0Eg4KBmRvY19pZBgBIAEoCRIRCgltYXhfc2NvcmUYAiABKAISLgoIc2VnbWVudHMYAyADKAsyHC50dXJiby5zZWFyY2gudjEuSGVhdFNlZ21lbnQiXwoURXh0cmFjdGVkRmlsdGVyUHJvdG8SDAoEa2luZBgBIAEoCRIPCgdzdW1tYXJ5GAIgASgJEigKBHNwYW4YAyABKAsyGi50dXJiby5zZWFyY2gudjEuU3BhblByb3RvIugBCg5TZWFyY2hSZXNwb25zZRIWCg5yZXNvbHZlZF9pbmRleBgBIAEoCRIoCgRoaXRzGAIgAygLMhoudHVyYm8uc2VhcmNoLnYxLlNlYXJjaEhpdBI0Cg1kb2N1bWVudF9oZWF0GAMgAygLMh0udHVyYm8uc2VhcmNoLnYxLkRvY3VtZW50SGVhdBIcChRhbmFseXplZF9xdWVyeV90ZXJtcxgEIAMoCRJAChFleHRyYWN0ZWRfZmlsdGVycxgFIAMoCzIlLnR1cmJvLnNlYXJjaC52MS5FeHRyYWN0ZWRGaWx0ZXJQcm90byI3ChNQZXJzaXN0SW5kZXhSZXF1ZXN0EhIKCmluZGV4X25hbWUYASABKAkSDAoEcGF0aBgCIAEoCSJBChRQZXJzaXN0SW5kZXhSZXNwb25zZRISCgppbmRleF9uYW1lGAEgASgJEhUKDWJ5dGVzX3dyaXR0ZW4YAiABKAQiJgoQU2VhbEluZGV4UmVxdWVzdBISCgppbmRleF9uYW1lGAEgASgJIkMKEVNlYWxJbmRleFJlc3BvbnNlEi4KB3N1bW1hcnkYASABKAsyHS50dXJiby5zZWFyY2gudjEuSW5kZXhTdW1tYXJ5ImsKDlJlaW5kZXhSZXF1ZXN0EhQKDHNvdXJjZV9pbmRleBgBIAEoCRIvCgt0YXJnZXRfc3BlYxgCIAEoCzIaLnR1cmJvLnNlYXJjaC52MS5JbmRleFNwZWMSEgoKc3dhcF9hbGlhcxgDIAEoCSJYCg9SZWluZGV4UmVzcG9uc2USLgoHc3VtbWFyeRgBIAEoCzIdLnR1cmJvLnNlYXJjaC52MS5JbmRleFN1bW1hcnkSFQoNc3dhcHBlZF9hbGlhcxgCIAEoCSI0Cg9TZXRBbGlhc1JlcXVlc3QSDQoFYWxpYXMYASABKAkSEgoKaW5kZXhfbmFtZRgCIAEoCSISChBTZXRBbGlhc1Jlc3BvbnNlIiMKEkRlbGV0ZUFsaWFzUmVxdWVzdBINCgVhbGlhcxgBIAEoCSIVChNEZWxldGVBbGlhc1Jlc3BvbnNlIi8KCkFsaWFzRW50cnkSDQoFYWxpYXMYASABKAkSEgoKaW5kZXhfbmFtZRgCIAEoCSIUChJMaXN0QWxpYXNlc1JlcXVlc3QiQwoTTGlzdEFsaWFzZXNSZXNwb25zZRIsCgdhbGlhc2VzGAEgAygLMhsudHVyYm8uc2VhcmNoLnYxLkFsaWFzRW50cnkiPgoSVGVybUZyZXF1ZW5jeUVudHJ5EgwKBHRlcm0YASABKAkSGgoSZG9jdW1lbnRfZnJlcXVlbmN5GAIgASgEImQKFFNldENvbGxlY3Rpb25SZXF1ZXN0EhcKD2NvbGxlY3Rpb25fbmFtZRgBIAEoCRIWCg5iYXNlbGluZV90ZXJtcxgCIAMoCRIbChNvb3ZfYWxlcnRfdGhyZXNob2xkGAMgASgBIkwKFVNldENvbGxlY3Rpb25SZXNwb25zZRIzCgpjb2xsZWN0aW9uGAEgASgLMh8udHVyYm8uc2VhcmNoLnYxLkNvbGxlY3Rpb25JbmZvIi8KFEdldENvbGxlY3Rpb25SZXF1ZXN0EhcKD2NvbGxlY3Rpb25fbmFtZRgBIAEoCSJMChVHZXRDb2xsZWN0aW9uUmVzcG9uc2USMwoKY29sbGVjdGlvbhgBIAEoCzIfLnR1cmJvLnNlYXJjaC52MS5Db2xsZWN0aW9uSW5mbyIYChZMaXN0Q29sbGVjdGlvbnNSZXF1ZXN0Ik8KF0xpc3RDb2xsZWN0aW9uc1Jlc3BvbnNlEjQKC2NvbGxlY3Rpb25zGAEgAygLMh8udHVyYm8uc2VhcmNoLnYxLkNvbGxlY3Rpb25JbmZvIjIKF0RlbGV0ZUNvbGxlY3Rpb25SZXF1ZXN0EhcKD2NvbGxlY3Rpb25fbmFtZRgBIAEoCSIaChhEZWxldGVDb2xsZWN0aW9uUmVzcG9uc2UiwQEKDkNvbGxlY3Rpb25JbmZvEhcKD2NvbGxlY3Rpb25fbmFtZRgBIAEoCRITCgt0b3RhbF91bml0cxgCIAEoBBITCgttZWFuX2xlbmd0aBgDIAEoARIXCg92b2NhYnVsYXJ5X3NpemUYBCABKAQSGwoTb292X2FsZXJ0X3RocmVzaG9sZBgFIAEoARI2Cgl0b3BfdGVybXMYBiADKAsyIy50dXJiby5zZWFyY2gudjEuVGVybUZyZXF1ZW5jeUVudHJ5IsABChVWb2NhYnVsYXJ5RHJpZnRSZXBvcnQSFwoPY29sbGVjdGlvbl9uYW1lGAEgASgJEhsKE3ZvY2FidWxhcnlfY292ZXJhZ2UYAiABKAESEQoJb292X3JhdGlvGAMgASgBEhEKCW5ld190ZXJtcxgEIAMoCRI2Cgl0b3BfdGVybXMYBSADKAsyIy50dXJiby5zZWFyY2gudjEuVGVybUZyZXF1ZW5jeUVudHJ5EhMKC3RvdGFsX3VuaXRzGAYgASgEIkgKFldhdGNoQ29sbGVjdGlvblJlcXVlc3QSFwoPY29sbGVjdGlvbl9uYW1lGAEgASgJEhUKDW1pbl9vb3ZfcmF0aW8YAiABKAEieAoPQ29sbGVjdGlvbkV2ZW50EhcKD2NvbGxlY3Rpb25fbmFtZRgBIAEoCRIUCgxzb3VyY2VfaW5kZXgYAiABKAkSNgoGcmVwb3J0GAMgASgLMiYudHVyYm8uc2VhcmNoLnYxLlZvY2FidWxhcnlEcmlmdFJlcG9ydCpaChFWZWN0b3JTdG9yYWdlS2luZBIgChxWRUNUT1JfU1RPUkFHRV9LSU5EX0ZMQVRfRjMyEAASIwofVkVDVE9SX1NUT1JBR0VfS0lORF9UVVJCT19RVUFOVBABKkUKDkRpc3RhbmNlTWV0cmljEhoKFkRJU1RBTkNFX01FVFJJQ19DT1NJTkUQABIXChNESVNUQU5DRV9NRVRSSUNfRE9UEAEqTQoRTm9ybWFsaXphdGlvbk1vZGUSGwoXTk9STUFMSVpBVElPTl9NT0RFX05PTkUQABIbChdOT1JNQUxJWkFUSU9OX01PREVfVU5JVBABKkEKDVRlcm1NYXRjaE1vZGUSFwoTVEVSTV9NQVRDSF9NT0RFX0FOWRAAEhcKE1RFUk1fTUFUQ0hfTU9ERV9BTEwQASo7CgxKb2luT3BlcmF0b3ISFAoQSk9JTl9PUEVSQVRPUl9PUhAAEhUKEUpPSU5fT1BFUkFUT1JfQU5EEAEqUAoORnVzaW9uU3RyYXRlZ3kSGQoVRlVTSU9OX1NUUkFURUdZX1NDT1JFEAASIwofRlVTSU9OX1NUUkFURUdZX1JFQ0lQUk9DQUxfUkFOSxABMtgMChJUdXJib1NlYXJjaFNlcnZpY2UScgoTTGlzdFNlYXJjaFByb3ZpZGVycxIrLnR1cmJvLnNlYXJjaC52MS5MaXN0U2VhcmNoUHJvdmlkZXJzUmVxdWVzdBosLnR1cmJvLnNlYXJjaC52MS5MaXN0U2VhcmNoUHJvdmlkZXJzUmVzcG9uc2UiABJaCgtMaXN0SW5kZXhlcxIjLnR1cmJvLnNlYXJjaC52MS5MaXN0SW5kZXhlc1JlcXVlc3QaJC50dXJiby5zZWFyY2gudjEuTGlzdEluZGV4ZXNSZXNwb25zZSIAEloKC0NyZWF0ZUluZGV4EiMudHVyYm8uc2VhcmNoLnYxLkNyZWF0ZUluZGV4UmVxdWVzdBokLnR1cmJvLnNlYXJjaC52MS5DcmVhdGVJbmRleFJlc3BvbnNlIgASWgoLRGVsZXRlSW5kZXgSIy50dXJiby5zZWFyY2gudjEuRGVsZXRlSW5kZXhSZXF1ZXN0GiQudHVyYm8uc2VhcmNoLnYxLkRlbGV0ZUluZGV4UmVzcG9uc2UiABJkCg5JbmRleERvY3VtZW50cxIlLnR1cmJvLnNlYXJjaC52MS5JbmRleERvY3VtZW50UmVxdWVzdBonLnR1cmJvLnNlYXJjaC52MS5JbmRleERvY3VtZW50c1Jlc3BvbnNlIgAoARJLCgZTZWFyY2gSHi50dXJiby5zZWFyY2gudjEuU2VhcmNoUmVxdWVzdBofLnR1cmJvLnNlYXJjaC52MS5TZWFyY2hSZXNwb25zZSIAEl0KDFBlcnNpc3RJbmRleBIkLnR1cmJvLnNlYXJjaC52MS5QZXJzaXN0SW5kZXhSZXF1ZXN0GiUudHVyYm8uc2VhcmNoLnYxLlBlcnNpc3RJbmRleFJlc3BvbnNlIgASVAoJU2VhbEluZGV4EiEudHVyYm8uc2VhcmNoLnYxLlNlYWxJbmRleFJlcXVlc3QaIi50dXJiby5zZWFyY2gudjEuU2VhbEluZGV4UmVzcG9uc2UiABJOCgdSZWluZGV4Eh8udHVyYm8uc2VhcmNoLnYxLlJlaW5kZXhSZXF1ZXN0GiAudHVyYm8uc2VhcmNoLnYxLlJlaW5kZXhSZXNwb25zZSIAElEKCFNldEFsaWFzEiAudHVyYm8uc2VhcmNoLnYxLlNldEFsaWFzUmVxdWVzdBohLnR1cmJvLnNlYXJjaC52MS5TZXRBbGlhc1Jlc3BvbnNlIgASWgoLRGVsZXRlQWxpYXMSIy50dXJiby5zZWFyY2gudjEuRGVsZXRlQWxpYXNSZXF1ZXN0GiQudHVyYm8uc2VhcmNoLnYxLkRlbGV0ZUFsaWFzUmVzcG9uc2UiABJaCgtMaXN0QWxpYXNlcxIjLnR1cmJvLnNlYXJjaC52MS5MaXN0QWxpYXNlc1JlcXVlc3QaJC50dXJiby5zZWFyY2gudjEuTGlzdEFsaWFzZXNSZXNwb25zZSIAEmAKDVNldENvbGxlY3Rpb24SJS50dXJiby5zZWFyY2gudjEuU2V0Q29sbGVjdGlvblJlcXVlc3QaJi50dXJiby5zZWFyY2gudjEuU2V0Q29sbGVjdGlvblJlc3BvbnNlIgASYAoNR2V0Q29sbGVjdGlvbhIlLnR1cmJvLnNlYXJjaC52MS5HZXRDb2xsZWN0aW9uUmVxdWVzdBomLnR1cmJvLnNlYXJjaC52MS5HZXRDb2xsZWN0aW9uUmVzcG9uc2UiABJmCg9MaXN0Q29sbGVjdGlvbnMSJy50dXJiby5zZWFyY2gudjEuTGlzdENvbGxlY3Rpb25zUmVxdWVzdBooLnR1cmJvLnNlYXJjaC52MS5MaXN0Q29sbGVjdGlvbnNSZXNwb25zZSIAEmkKEERlbGV0ZUNvbGxlY3Rpb24SKC50dXJiby5zZWFyY2gudjEuRGVsZXRlQ29sbGVjdGlvblJlcXVlc3QaKS50dXJiby5zZWFyY2gudjEuRGVsZXRlQ29sbGVjdGlvblJlc3BvbnNlIgASYAoPV2F0Y2hDb2xsZWN0aW9uEicudHVyYm8uc2VhcmNoLnYxLldhdGNoQ29sbGVjdGlvblJlcXVlc3QaIC50dXJiby5zZWFyY2gudjEuQ29sbGVjdGlvbkV2ZW50IgAwAWIGcHJvdG8z");
var IndexDocumentRequestSchema = /* @__PURE__ */ messageDesc(file_turbo_search, 18);
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
var file_turbo_training = /* @__PURE__ */ fileDesc("ChR0dXJib190cmFpbmluZy5wcm90bxIRdHVyYm8udHJhaW5pbmcudjEiXgoLTW9kZWxTb3VyY2USDAoEbmFtZRgBIAEoCRIPCgdhZGRyZXNzGAIgASgJEjAKB3NpZ25faW4YAyABKA4yHy50dXJiby50cmFpbmluZy52MS5Tb3VyY2VTaWduSW4iJwoXTGlzdE1vZGVsU291cmNlc1JlcXVlc3QSDAoEdXNlchgBIAEoCSJ/ChhMaXN0TW9kZWxTb3VyY2VzUmVzcG9uc2USDAoEdXNlchgBIAEoCRIvCgdzb3VyY2VzGAIgAygLMh4udHVyYm8udHJhaW5pbmcudjEuTW9kZWxTb3VyY2USDQoFYWRtaW4YAyABKAgSFQoNZHJpdmVfc2lnbl9pbhgEIAEoCCKGAQoaUmVnaXN0ZXJNb2RlbFNvdXJjZVJlcXVlc3QSDAoEdXNlchgBIAEoCRIMCgRuYW1lGAIgASgJEg8KB2FkZHJlc3MYAyABKAkSDwoHYXBpX2tleRgEIAEoCRIqCgdzM19rZXlzGAUgASgLMhkudHVyYm8udHJhaW5pbmcudjEuUzNLZXlzInMKBlMzS2V5cxIVCg1hY2Nlc3Nfa2V5X2lkGAEgASgJEhkKEXNlY3JldF9hY2Nlc3Nfa2V5GAIgASgJEhUKDXNlc3Npb25fdG9rZW4YAyABKAkSDgoGcmVnaW9uGAQgASgJEhAKCGVuZHBvaW50GAUgASgJIlsKG1JlZ2lzdGVyTW9kZWxTb3VyY2VSZXNwb25zZRIMCgR1c2VyGAEgASgJEi4KBnNvdXJjZRgCIAEoCzIeLnR1cmJvLnRyYWluaW5nLnYxLk1vZGVsU291cmNlIjYKGFJlbW92ZU1vZGVsU291cmNlUmVxdWVzdBIMCgR1c2VyGAEgASgJEgwKBG5hbWUYAiABKAkiQQoZUmVtb3ZlTW9kZWxTb3VyY2VSZXNwb25zZRIMCgR1c2VyGAEgASgJEhYKDndhc19yZWdpc3RlcmVkGAIgASgIIkYKF1N0YXJ0RHJpdmVTaWduSW5SZXF1ZXN0EgwKBHVzZXIYASABKAkSDAoEbmFtZRgCIAEoCRIPCgdhZGRyZXNzGAMgASgJIlQKGFN0YXJ0RHJpdmVTaWduSW5SZXNwb25zZRIVCg1hdXRob3JpemVfdXJsGAEgASgJEg0KBXN0YXRlGAIgASgJEhIKCmV4cGlyZXNfaW4YAyABKA0iRQoYRmluaXNoRHJpdmVTaWduSW5SZXF1ZXN0EgwKBHVzZXIYASABKAkSDQoFc3RhdGUYAiABKAkSDAoEY29kZRgDIAEoCSJZChlGaW5pc2hEcml2ZVNpZ25JblJlc3BvbnNlEgwKBHVzZXIYASABKAkSLgoGc291cmNlGAIgASgLMh4udHVyYm8udHJhaW5pbmcudjEuTW9kZWxTb3VyY2UiLQoMTGljZW5jZVRlcm1zEg8KB2xpY2VuY2UYASABKAkSDAoEdGV4dBgCIAEoCSJGCg5UcmFpbmluZ1NvdXJjZRIMCgRuYW1lGAEgASgJEg8KB2xpY2VuY2UYAiABKAkSFQoNbGljZW5jZV90ZXJtcxgDIAEoCSKqAQoLQ2F0YWxvZ0ZpbGUSDAoEcGF0aBgBIAEoCRILCgN1cmwYAiABKAkSDAoEc2l6ZRgDIAEoBBIOCgZzaGEyNTYYBCABKAkSDgoGc2hhNTEyGAUgASgJEhMKC2NvbXByZXNzaW9uGAYgASgJEhYKDmRvd25sb2FkX2J5dGVzGAcgASgEEhcKD2Rvd25sb2FkX3NoYTI1NhgIIAEoCRIMCgRhbHNvGAkgAygJIrgDCgxDYXRhbG9nRW50cnkSCgoCaWQYASABKAkSDAoEa2luZBgCIAEoCRIRCgljb21wb25lbnQYAyABKAkSEAoIbGFuZ3VhZ2UYBCABKAkSDgoGZm9ybWF0GAUgASgJEg8KB3ZlcnNpb24YBiABKAkSDwoHbGljZW5jZRgHIAEoCRIVCg1saWNlbmNlX3Rlcm1zGAggASgJEhUKDWxpY2VuY2VfY2xhc3MYCSABKAkSEwoLYXR0cmlidXRpb24YCiABKAkSNQoKdHJhaW5lZF9vbhgLIAMoCzIhLnR1cmJvLnRyYWluaW5nLnYxLlRyYWluaW5nU291cmNlEi0KBWZpbGVzGAwgAygLMh4udHVyYm8udHJhaW5pbmcudjEuQ2F0YWxvZ0ZpbGUSEwoLdG90YWxfYnl0ZXMYDSABKAQSLgoFdGVybXMYDiADKAsyHy50dXJiby50cmFpbmluZy52MS5MaWNlbmNlVGVybXMSNgoNbWlzc2luZ190ZXJtcxgPIAMoCzIfLnR1cmJvLnRyYWluaW5nLnYxLkxpY2VuY2VUZXJtcxIRCglpbnN0YWxsZWQYECABKAgiRwoXTGlzdE1vZGVsQ2F0YWxvZ1JlcXVlc3QSDAoEdXNlchgBIAEoCRIMCgRraW5kGAIgASgJEhAKCGxhbmd1YWdlGAMgASgJIkwKGExpc3RNb2RlbENhdGFsb2dSZXNwb25zZRIwCgdlbnRyaWVzGAEgAygLMh8udHVyYm8udHJhaW5pbmcudjEuQ2F0YWxvZ0VudHJ5Il4KDkluc3RhbGxlZE1vZGVsEgoKAmlkGAEgASgJEg8KB3ZlcnNpb24YAiABKAkSCwoDZGlyGAMgASgJEg0KBWZpbGVzGAQgAygJEhMKC3RvdGFsX2J5dGVzGAUgASgEIhwKGkxpc3RJbnN0YWxsZWRNb2RlbHNSZXF1ZXN0IlAKG0xpc3RJbnN0YWxsZWRNb2RlbHNSZXNwb25zZRIxCgZtb2RlbHMYASADKAsyIS50dXJiby50cmFpbmluZy52MS5JbnN0YWxsZWRNb2RlbCJ4Ch5SZWNvcmRMaWNlbmNlQWNjZXB0YW5jZVJlcXVlc3QSDAoEdXNlchgBIAEoCRIPCgdsaWNlbmNlGAIgASgJEg0KBXRlcm1zGAMgASgJEhMKC2FydGlmYWN0X2lkGAQgASgJEhMKC3RleHRfc2hhMjU2GAUgASgJIncKH1JlY29yZExpY2VuY2VBY2NlcHRhbmNlUmVzcG9uc2USDAoEdXNlchgBIAEoCRIxCghhY2NlcHRlZBgCIAEoCzIfLnR1cmJvLnRyYWluaW5nLnYxLkxpY2VuY2VUZXJtcxITCgthY2NlcHRlZF9hdBgDIAEoBCJOCh5SZXZva2VMaWNlbmNlQWNjZXB0YW5jZVJlcXVlc3QSDAoEdXNlchgBIAEoCRIPCgdsaWNlbmNlGAIgASgJEg0KBXRlcm1zGAMgASgJIncKH1Jldm9rZUxpY2VuY2VBY2NlcHRhbmNlUmVzcG9uc2USDAoEdXNlchgBIAEoCRIwCgdyZXZva2VkGAIgASgLMh8udHVyYm8udHJhaW5pbmcudjEuTGljZW5jZVRlcm1zEhQKDHdhc19hY2NlcHRlZBgDIAEoCCKCAQoRTGljZW5jZUFjY2VwdGFuY2USLgoFdGVybXMYASABKAsyHy50dXJiby50cmFpbmluZy52MS5MaWNlbmNlVGVybXMSEwoLYWNjZXB0ZWRfYXQYAiABKAQSEwoLYXJ0aWZhY3RfaWQYAyABKAkSEwoLdGV4dF9zaGEyNTYYBCABKAkiLQodTGlzdExpY2VuY2VBY2NlcHRhbmNlc1JlcXVlc3QSDAoEdXNlchgBIAEoCSJ1Ch5MaXN0TGljZW5jZUFjY2VwdGFuY2VzUmVzcG9uc2USDAoEdXNlchgBIAEoCRI2CghhY2NlcHRlZBgCIAMoCzIkLnR1cmJvLnRyYWluaW5nLnYxLkxpY2VuY2VBY2NlcHRhbmNlEg0KBWFkbWluGAMgASgIIjgKE0luc3RhbGxNb2RlbFJlcXVlc3QSDAoEdXNlchgBIAEoCRITCgthcnRpZmFjdF9pZBgCIAEoCSKGAQoPSW5zdGFsbFByb2dyZXNzEi4KBXN0YWdlGAEgASgOMh8udHVyYm8udHJhaW5pbmcudjEuSW5zdGFsbFN0YWdlEhIKCmZpbGVfaW5kZXgYAiABKA0SEgoKZmlsZV9jb3VudBgDIAEoDRIMCgRwYXRoGAQgASgJEg0KBWJ5dGVzGAUgASgEIo4BChJJbnN0YWxsTW9kZWxVcGRhdGUSNgoIcHJvZ3Jlc3MYASABKAsyIi50dXJiby50cmFpbmluZy52MS5JbnN0YWxsUHJvZ3Jlc3NIABI2CglpbnN0YWxsZWQYAiABKAsyIS50dXJiby50cmFpbmluZy52MS5JbnN0YWxsZWRNb2RlbEgAQggKBnVwZGF0ZSI5ChREZXNjcmliZU1vZGVsUmVxdWVzdBIMCgR1c2VyGAEgASgJEhMKC2FydGlmYWN0X2lkGAIgASgJIkcKFURlc2NyaWJlTW9kZWxSZXNwb25zZRIuCgVlbnRyeRgBIAEoCzIfLnR1cmJvLnRyYWluaW5nLnYxLkNhdGFsb2dFbnRyeSJIChNSZXNvbHZlTW9kZWxSZXF1ZXN0Eg4KBnNoYTI1NhgBIAEoCRITCgthcnRpZmFjdF9pZBgCIAEoCRIMCgR1c2VyGAMgASgJIlQKFFJlc29sdmVNb2RlbFJlc3BvbnNlEi4KBWVudHJ5GAEgASgLMh8udHVyYm8udHJhaW5pbmcudjEuQ2F0YWxvZ0VudHJ5EgwKBHBhdGgYAiABKAkiVwoUUmVhZE1vZGVsRmlsZVJlcXVlc3QSDAoEdXNlchgBIAEoCRITCgthcnRpZmFjdF9pZBgCIAEoCRIMCgRwYXRoGAMgASgJEg4KBnNoYTI1NhgEIAEoCSJZCg5Nb2RlbEZpbGVDaHVuaxIuCgRmaWxlGAEgASgLMh4udHVyYm8udHJhaW5pbmcudjEuQ2F0YWxvZ0ZpbGVIABIOCgRkYXRhGAIgASgMSABCBwoFZnJhbWUieAoHVGVhY2hlchIMCgRuYW1lGAEgASgJEhAKCG1vZGVsX2lkGAIgASgJEhAKCHJldmlzaW9uGAMgASgJEhcKD21hbmlmZXN0X3NoYTI1NhgEIAEoCRIOCgZoaWRkZW4YBSABKA0SEgoKdm9jYWJfc2l6ZRgGIAEoDSIVChNMaXN0VGVhY2hlcnNSZXF1ZXN0IkQKFExpc3RUZWFjaGVyc1Jlc3BvbnNlEiwKCHRlYWNoZXJzGAEgAygLMhoudHVyYm8udHJhaW5pbmcudjEuVGVhY2hlciL5AQoXVHJhaW5TdGF0aWNNb2RlbFJlcXVlc3QSDAoEbmFtZRgBIAEoCRIPCgd0ZWFjaGVyGAIgASgJEhAKCHBjYV9kaW1zGAMgASgNEhcKD3NpZl9jb2VmZmljaWVudBgEIAEoARItCgVkdHlwZRgFIAEoDjIeLnR1cmJvLnRyYWluaW5nLnYxLlN0YXRpY0R0eXBlEhIKCm1heF9sZW5ndGgYBiABKA0SHgoWdm9jYWJ1bGFyeV9hcnRpZmFjdF9pZBgHIAEoCRINCgV0ZXJtcxgIIAMoCRIQCghyZXZpc2lvbhgJIAEoCRIQCghzZXNzaW9ucxgKIAEoDSL2AQoYVHJhaW5TdGF0aWNNb2RlbFByb2dyZXNzEiwKBXN0YWdlGAEgASgOMh0udHVyYm8udHJhaW5pbmcudjEuVHJhaW5TdGFnZRIXCg9jb21wbGV0ZWRfaXRlbXMYAiABKAQSEwoLdG90YWxfaXRlbXMYAyABKAQSEwoLYWRkZWRfdGVybXMYBCABKAQSFQoNcmVmdXNlZF90ZXJtcxgFIAEoBBIaChJleHBsYWluZWRfdmFyaWFuY2UYBiABKAESNgoHcXVhbGl0eRgHIAEoCzIlLnR1cmJvLnRyYWluaW5nLnYxLlN0YXRpY01vZGVsUXVhbGl0eSJQChFUZWFjaGVyUHJvdmVuYW5jZRIQCghtb2RlbF9pZBgBIAEoCRIQCghyZXZpc2lvbhgCIAEoCRIXCg9tYW5pZmVzdF9zaGEyNTYYAyABKAkiWQoSU3RhdGljTW9kZWxRdWFsaXR5EhEKCWJhc2VfdG9wMRgBIAEoARITCgtzdGF0aWNfdG9wMRgCIAEoARIbChNzaW1pbGFyaXR5X3NwZWFybWFuGAMgASgBIjoKDEFydGlmYWN0RmlsZRIMCgRwYXRoGAEgASgJEgwKBHNpemUYAiABKAQSDgoGc2hhMjU2GAMgASgJIvwCChVTdGF0aWNNb2RlbERlc2NyaXB0b3ISDAoEbmFtZRgBIAEoCRIQCghtb2RlbF9pZBgCIAEoCRIQCghyZXZpc2lvbhgDIAEoCRIXCg9tYW5pZmVzdF9zaGEyNTYYBCABKAkSDAoEZGltcxgFIAEoDRISCgp2b2NhYl9zaXplGAYgASgNEhUKDWxlYXJuZWRfdGVybXMYByABKA0SDQoFZHR5cGUYCCABKAkSNQoHdGVhY2hlchgJIAEoCzIkLnR1cmJvLnRyYWluaW5nLnYxLlRlYWNoZXJQcm92ZW5hbmNlEjYKB3F1YWxpdHkYCiABKAsyJS50dXJiby50cmFpbmluZy52MS5TdGF0aWNNb2RlbFF1YWxpdHkSLgoFZmlsZXMYCyADKAsyHy50dXJiby50cmFpbmluZy52MS5BcnRpZmFjdEZpbGUSEAoIdmVyaWZpZWQYDCABKAgSDwoHcHJvYmxlbRgNIAEoCRIOCgZzZXJ2ZWQYDiABKAgiKwoLUmVmdXNlZFRlcm0SDAoEdGVybRgBIAEoCRIOCgZyZWFzb24YAiABKAki1AEKElRyYWluZWRTdGF0aWNNb2RlbBI3CgVtb2RlbBgBIAEoCzIoLnR1cmJvLnRyYWluaW5nLnYxLlN0YXRpY01vZGVsRGVzY3JpcHRvchITCgthZGRlZF90ZXJtcxgCIAMoCRI1Cg1yZWZ1c2VkX3Rlcm1zGAMgAygLMh4udHVyYm8udHJhaW5pbmcudjEuUmVmdXNlZFRlcm0SHgoWdm9jYWJ1bGFyeV9hcnRpZmFjdF9pZBgEIAEoCRIZChF2b2NhYnVsYXJ5X3NoYTI1NhgFIAEoCSKdAQoWVHJhaW5TdGF0aWNNb2RlbFVwZGF0ZRI/Cghwcm9ncmVzcxgBIAEoCzIrLnR1cmJvLnRyYWluaW5nLnYxLlRyYWluU3RhdGljTW9kZWxQcm9ncmVzc0gAEjgKB3RyYWluZWQYAiABKAsyJS50dXJiby50cmFpbmluZy52MS5UcmFpbmVkU3RhdGljTW9kZWxIAEIICgZ1cGRhdGUiGQoXTGlzdFN0YXRpY01vZGVsc1JlcXVlc3QiVAoYTGlzdFN0YXRpY01vZGVsc1Jlc3BvbnNlEjgKBm1vZGVscxgBIAMoCzIoLnR1cmJvLnRyYWluaW5nLnYxLlN0YXRpY01vZGVsRGVzY3JpcHRvciIoChhEZWxldGVTdGF0aWNNb2RlbFJlcXVlc3QSDAoEbmFtZRgBIAEoCSItChlEZWxldGVTdGF0aWNNb2RlbFJlc3BvbnNlEhAKCHVubG9hZGVkGAEgASgIIu4BChNUcmFpbkNvbXBvbmVudFN0YXJ0EjMKCWNvbXBvbmVudBgBIAEoDjIgLnR1cmJvLnRyYWluaW5nLnYxLkNvbXBvbmVudEtpbmQSEAoIbGFuZ3VhZ2UYAiABKAkSSgoKcGFyYW1ldGVycxgDIAMoCzI2LnR1cmJvLnRyYWluaW5nLnYxLlRyYWluQ29tcG9uZW50U3RhcnQuUGFyYW1ldGVyc0VudHJ5EhEKCW5hbWVfdHlwZRgEIAEoCRoxCg9QYXJhbWV0ZXJzRW50cnkSCwoDa2V5GAEgASgJEg0KBXZhbHVlGAIgASgJOgI4ASJrChVUcmFpbkNvbXBvbmVudFJlcXVlc3QSNwoFc3RhcnQYASABKAsyJi50dXJiby50cmFpbmluZy52MS5UcmFpbkNvbXBvbmVudFN0YXJ0SAASEQoHc2FtcGxlcxgCIAEoDEgAQgYKBHBhcnQirwEKFlRyYWluQ29tcG9uZW50UHJvZ3Jlc3MSMAoFc3RhZ2UYASABKA4yIS50dXJiby50cmFpbmluZy52MS5Db21wb25lbnRTdGFnZRIPCgdzYW1wbGVzGAIgASgEEhEKCWl0ZXJhdGlvbhgDIAEoDRIPCgdjb3JyZWN0GAQgASgEEg4KBmV2ZW50cxgFIAEoBBIPCgdtZWFzdXJlGAYgASgJEg0KBXZhbHVlGAcgASgBInYKEFRyYWluZWRDb21wb25lbnQSMwoJY29tcG9uZW50GAEgASgOMiAudHVyYm8udHJhaW5pbmcudjEuQ29tcG9uZW50S2luZBIPCgdzYW1wbGVzGAIgASgEEgwKBHNpemUYAyABKAQSDgoGc2hhMjU2GAQgASgJIqgBChRUcmFpbkNvbXBvbmVudFVwZGF0ZRI9Cghwcm9ncmVzcxgBIAEoCzIpLnR1cmJvLnRyYWluaW5nLnYxLlRyYWluQ29tcG9uZW50UHJvZ3Jlc3NIABIPCgVtb2RlbBgCIAEoDEgAEjYKB3RyYWluZWQYAyABKAsyIy50dXJiby50cmFpbmluZy52MS5UcmFpbmVkQ29tcG9uZW50SABCCAoGdXBkYXRlIqgBChhTdHJlYW1pbmdUcmFpbmluZ1JlcXVlc3QSOgoFc3RhcnQYASABKAsyKS50dXJiby50cmFpbmluZy52MS5TdHJlYW1pbmdUcmFpbmluZ1N0YXJ0SAASNQoIZG9jdW1lbnQYAiABKAsyIS50dXJiby50cmFpbmluZy52MS5Db3JwdXNEb2N1bWVudEgAEhEKB3NhbXBsZXMYAyABKAxIAEIGCgRwYXJ0IpUBChZTdHJlYW1pbmdUcmFpbmluZ1N0YXJ0EjMKBmNvcnB1cxgBIAEoCzIhLnR1cmJvLnRyYWluaW5nLnYxLkNvcnB1c1RyYWluaW5nSAASPwoJY29tcG9uZW50GAIgASgLMioudHVyYm8udHJhaW5pbmcudjEuTmV1cmFsQ29tcG9uZW50VHJhaW5pbmdIAEIFCgNqb2IiLgoOQ29ycHVzRG9jdW1lbnQSDgoGZG9jX2lkGAEgASgJEgwKBHRleHQYAiABKAkiswEKDkNvcnB1c1RyYWluaW5nEjcKCnZvY2FidWxhcnkYASABKAsyIy50dXJiby50cmFpbmluZy52MS5Db3JwdXNWb2NhYnVsYXJ5EjkKBW1vZGVsGAIgASgLMioudHVyYm8udHJhaW5pbmcudjEuVHJhaW5TdGF0aWNNb2RlbFJlcXVlc3QSLQoFaW5kZXgYAyABKAsyHi50dXJiby50cmFpbmluZy52MS5Db3JwdXNJbmRleCL5AQoQQ29ycHVzVm9jYWJ1bGFyeRIaChJtaW5fdGVybV9mcmVxdWVuY3kYASABKA0SEQoJbWF4X3Rlcm1zGAIgASgNEhEKCWxvd2VyY2FzZRgDIAEoCBIYChBmaWx0ZXJfc3RvcHdvcmRzGAQgASgIEhYKDmluY2x1ZGVfbmdyYW1zGAUgASgIEhYKDm1pbl9uZ3JhbV9zaXplGAYgASgNEhYKDm1heF9uZ3JhbV9zaXplGAcgASgNEg8KB21pbl9wbWkYCCABKAESHgoWZGljdGlvbmFyeV9hcnRpZmFjdF9pZBgJIAEoCRIQCghsYW5ndWFnZRgKIAEoCSK8AQoLQ29ycHVzSW5kZXgSEgoKaW5kZXhfbmFtZRgBIAEoCRI2CgdzdG9yYWdlGAIgASgOMiUudHVyYm8udHJhaW5pbmcudjEuQ29ycHVzSW5kZXhTdG9yYWdlEhIKCnF1YW50X2JpdHMYAyABKA0SFwoPcXVhbnRfY2FsaWJyYXRlGAQgASgIEhcKD2NvbGxlY3Rpb25fbmFtZRgFIAEoCRIbChNjaHVua19idWRnZXRfdG9rZW5zGAYgASgNIuYDChdOZXVyYWxDb21wb25lbnRUcmFpbmluZxI5Cgljb21wb25lbnQYASABKA4yJi50dXJiby50cmFpbmluZy52MS5OZXVyYWxDb21wb25lbnRLaW5kEi8KBnRhZ3NldBgCIAEoDjIfLnR1cmJvLnRyYWluaW5nLnYxLkNvbmxsdVRhZ3NldBIWCg5lbWJlZGRpbmdfc2l6ZRgDIAEoDRITCgtoaWRkZW5fc2l6ZRgEIAEoDRIOCgZlcG9jaHMYBSABKA0SEgoKYmF0Y2hfc2l6ZRgGIAEoDRIVCg1sZWFybmluZ19yYXRlGAcgASgBEg8KAmwyGAggASgBSACIAQESFAoHZHJvcG91dBgJIAEoAUgBiAEBEhgKC3dvcmRfY3V0b2ZmGAogASgNSAKIAQESEQoEc2VlZBgLIAEoA0gDiAEBEjoKBmxheW91dBgMIAEoDjIqLnR1cmJvLnRyYWluaW5nLnYxLkRlcGVuZGVuY3lGZWF0dXJlTGF5b3V0EjsKCmFjdGl2YXRpb24YDSABKA4yJy50dXJiby50cmFpbmluZy52MS5EZXBlbmRlbmN5QWN0aXZhdGlvbkIFCgNfbDJCCgoIX2Ryb3BvdXRCDgoMX3dvcmRfY3V0b2ZmQgcKBV9zZWVkIswBChlTdHJlYW1pbmdUcmFpbmluZ1Byb2dyZXNzEjAKBXN0YWdlGAEgASgOMiEudHVyYm8udHJhaW5pbmcudjEuU3RyZWFtaW5nU3RhZ2USFwoPY29tcGxldGVkX2l0ZW1zGAIgASgEEhMKC3RvdGFsX2l0ZW1zGAMgASgEEkEKDGRpc3RpbGxhdGlvbhgEIAEoCzIrLnR1cmJvLnRyYWluaW5nLnYxLlRyYWluU3RhdGljTW9kZWxQcm9ncmVzcxIMCgRsb3NzGAUgASgBIpgCCg1Db3JwdXNUcmFpbmVkEh4KFnZvY2FidWxhcnlfYXJ0aWZhY3RfaWQYASABKAkSGQoRdm9jYWJ1bGFyeV9zaGEyNTYYAiABKAkSGgoSdm9jYWJ1bGFyeV9jcmVhdGVkGAMgASgIEhgKEHZvY2FidWxhcnlfdGVybXMYBCABKAQSGQoRdm9jYWJ1bGFyeV9uZ3JhbXMYBSABKAQSNAoFbW9kZWwYBiABKAsyJS50dXJiby50cmFpbmluZy52MS5UcmFpbmVkU3RhdGljTW9kZWwSEgoKaW5kZXhfbmFtZRgHIAEoCRIZChFpbmRleGVkX2RvY3VtZW50cxgIIAEoBBIWCg5pbmRleGVkX2NodW5rcxgJIAEoBCKCAQoWVHJhaW5lZE5ldXJhbENvbXBvbmVudBI5Cgljb21wb25lbnQYASABKA4yJi50dXJiby50cmFpbmluZy52MS5OZXVyYWxDb21wb25lbnRLaW5kEg8KB3NhbXBsZXMYAiABKAQSDAoEc2l6ZRgDIAEoBBIOCgZzaGEyNTYYBCABKAki6gEKF1N0cmVhbWluZ1RyYWluaW5nVXBkYXRlEkAKCHByb2dyZXNzGAEgASgLMiwudHVyYm8udHJhaW5pbmcudjEuU3RyZWFtaW5nVHJhaW5pbmdQcm9ncmVzc0gAEg8KBW1vZGVsGAIgASgMSAASMgoGY29ycHVzGAMgASgLMiAudHVyYm8udHJhaW5pbmcudjEuQ29ycHVzVHJhaW5lZEgAEj4KCWNvbXBvbmVudBgEIAEoCzIpLnR1cmJvLnRyYWluaW5nLnYxLlRyYWluZWROZXVyYWxDb21wb25lbnRIAEIICgZ1cGRhdGUqoQEKDFNvdXJjZVNpZ25JbhIeChpTT1VSQ0VfU0lHTl9JTl9VTlNQRUNJRklFRBAAEh4KGlNPVVJDRV9TSUdOX0lOX0VOVklST05NRU5UEAESGQoVU09VUkNFX1NJR05fSU5fR09PR0xFEAISGgoWU09VUkNFX1NJR05fSU5fQVBJX0tFWRADEhoKFlNPVVJDRV9TSUdOX0lOX1MzX0tFWVMQBCqiAQoMSW5zdGFsbFN0YWdlEh0KGUlOU1RBTExfU1RBR0VfVU5TUEVDSUZJRUQQABIaChZJTlNUQUxMX1NUQUdFX0FETUlUVEVEEAESHQoZSU5TVEFMTF9TVEFHRV9ET1dOTE9BRElORxACEhwKGElOU1RBTExfU1RBR0VfRE9XTkxPQURFRBADEhoKFklOU1RBTExfU1RBR0VfVkVSSUZJRUQQBCpXCgtTdGF0aWNEdHlwZRIcChhTVEFUSUNfRFRZUEVfVU5TUEVDSUZJRUQQABIUChBTVEFUSUNfRFRZUEVfRjMyEAESFAoQU1RBVElDX0RUWVBFX0YxNhACKtcBCgpUcmFpblN0YWdlEhsKF1RSQUlOX1NUQUdFX1VOU1BFQ0lGSUVEEAASGAoUVFJBSU5fU1RBR0VfQURNSVRURUQQARIUChBUUkFJTl9TVEFHRV9LRVBUEAISGQoVVFJBSU5fU1RBR0VfRU1CRURESU5HEAMSFwoTVFJBSU5fU1RBR0VfUkVEVUNFRBAEEhgKFFRSQUlOX1NUQUdFX01FQVNVUkVEEAUSFgoSVFJBSU5fU1RBR0VfU0VBTEVEEAYSFgoSVFJBSU5fU1RBR0VfTE9BREVEEAcqvAIKDUNvbXBvbmVudEtpbmQSHgoaQ09NUE9ORU5UX0tJTkRfVU5TUEVDSUZJRUQQABIkCiBDT01QT05FTlRfS0lORF9TRU5URU5DRV9ERVRFQ1RPUhABEhwKGENPTVBPTkVOVF9LSU5EX1RPS0VOSVpFUhACEh0KGUNPTVBPTkVOVF9LSU5EX1BPU19UQUdHRVIQAxIdChlDT01QT05FTlRfS0lORF9MRU1NQVRJWkVSEAQSGgoWQ09NUE9ORU5UX0tJTkRfQ0hVTktFUhAFEh4KGkNPTVBPTkVOVF9LSU5EX05BTUVfRklOREVSEAYSJwojQ09NUE9ORU5UX0tJTkRfRE9DVU1FTlRfQ0FURUdPUklaRVIQBxIkCiBDT01QT05FTlRfS0lORF9MQU5HVUFHRV9ERVRFQ1RPUhAIKqYBCg5Db21wb25lbnRTdGFnZRIfChtDT01QT05FTlRfU1RBR0VfVU5TUEVDSUZJRUQQABIcChhDT01QT05FTlRfU1RBR0VfQURNSVRURUQQARIYChRDT01QT05FTlRfU1RBR0VfUkVBRBACEh0KGUNPTVBPTkVOVF9TVEFHRV9JVEVSQVRJT04QAxIcChhDT01QT05FTlRfU1RBR0VfUEFDS0FHRUQQBCpdChJDb3JwdXNJbmRleFN0b3JhZ2USIQodQ09SUFVTX0lOREVYX1NUT1JBR0VfRkxBVF9GMzIQABIkCiBDT1JQVVNfSU5ERVhfU1RPUkFHRV9UVVJCT19RVUFOVBABKmkKE05ldXJhbENvbXBvbmVudEtpbmQSJQohTkVVUkFMX0NPTVBPTkVOVF9LSU5EX1VOU1BFQ0lGSUVEEAASKwonTkVVUkFMX0NPTVBPTkVOVF9LSU5EX0RFUEVOREVOQ1lfUEFSU0VSEAEqPgoMQ29ubGx1VGFnc2V0EhYKEkNPTkxMVV9UQUdTRVRfVVBPUxAAEhYKEkNPTkxMVV9UQUdTRVRfWFBPUxABKnIKF0RlcGVuZGVuY3lGZWF0dXJlTGF5b3V0Ei0KKURFUEVOREVOQ1lfRkVBVFVSRV9MQVlPVVRfQ0hFTl9NQU5OSU5HXzE4EAASKAokREVQRU5ERU5DWV9GRUFUVVJFX0xBWU9VVF9DT01QQUNUXzE0EAEqVgoURGVwZW5kZW5jeUFjdGl2YXRpb24SHgoaREVQRU5ERU5DWV9BQ1RJVkFUSU9OX0NVQkUQABIeChpERVBFTkRFTkNZX0FDVElWQVRJT05fUkVMVRABKs0CCg5TdHJlYW1pbmdTdGFnZRIfChtTVFJFQU1JTkdfU1RBR0VfVU5TUEVDSUZJRUQQABIcChhTVFJFQU1JTkdfU1RBR0VfQURNSVRURUQQARIdChlTVFJFQU1JTkdfU1RBR0VfQU5BTFlaSU5HEAISJwojU1RSRUFNSU5HX1NUQUdFX0xFQVJOSU5HX1ZPQ0FCVUxBUlkQAxIkCiBTVFJFQU1JTkdfU1RBR0VfRElTVElMTElOR19NT0RFTBAEEhwKGFNUUkVBTUlOR19TVEFHRV9JTkRFWElORxAFEh0KGVNUUkVBTUlOR19TVEFHRV9DT01QTEVURUQQBhIYChRTVFJFQU1JTkdfU1RBR0VfUkVBRBAHEhkKFVNUUkVBTUlOR19TVEFHRV9FUE9DSBAIEhwKGFNUUkVBTUlOR19TVEFHRV9QQUNLQUdFRBAJMrIMChNUdXJib0NhdGFsb2dTZXJ2aWNlEmsKEExpc3RNb2RlbENhdGFsb2cSKi50dXJiby50cmFpbmluZy52MS5MaXN0TW9kZWxDYXRhbG9nUmVxdWVzdBorLnR1cmJvLnRyYWluaW5nLnYxLkxpc3RNb2RlbENhdGFsb2dSZXNwb25zZRJ0ChNMaXN0SW5zdGFsbGVkTW9kZWxzEi0udHVyYm8udHJhaW5pbmcudjEuTGlzdEluc3RhbGxlZE1vZGVsc1JlcXVlc3QaLi50dXJiby50cmFpbmluZy52MS5MaXN0SW5zdGFsbGVkTW9kZWxzUmVzcG9uc2USgAEKF1JlY29yZExpY2VuY2VBY2NlcHRhbmNlEjEudHVyYm8udHJhaW5pbmcudjEuUmVjb3JkTGljZW5jZUFjY2VwdGFuY2VSZXF1ZXN0GjIudHVyYm8udHJhaW5pbmcudjEuUmVjb3JkTGljZW5jZUFjY2VwdGFuY2VSZXNwb25zZRJfCgxJbnN0YWxsTW9kZWwSJi50dXJiby50cmFpbmluZy52MS5JbnN0YWxsTW9kZWxSZXF1ZXN0GiUudHVyYm8udHJhaW5pbmcudjEuSW5zdGFsbE1vZGVsVXBkYXRlMAESYgoNRGVzY3JpYmVNb2RlbBInLnR1cmJvLnRyYWluaW5nLnYxLkRlc2NyaWJlTW9kZWxSZXF1ZXN0GigudHVyYm8udHJhaW5pbmcudjEuRGVzY3JpYmVNb2RlbFJlc3BvbnNlEl8KDFJlc29sdmVNb2RlbBImLnR1cmJvLnRyYWluaW5nLnYxLlJlc29sdmVNb2RlbFJlcXVlc3QaJy50dXJiby50cmFpbmluZy52MS5SZXNvbHZlTW9kZWxSZXNwb25zZRJdCg1SZWFkTW9kZWxGaWxlEicudHVyYm8udHJhaW5pbmcudjEuUmVhZE1vZGVsRmlsZVJlcXVlc3QaIS50dXJiby50cmFpbmluZy52MS5Nb2RlbEZpbGVDaHVuazABEoABChdSZXZva2VMaWNlbmNlQWNjZXB0YW5jZRIxLnR1cmJvLnRyYWluaW5nLnYxLlJldm9rZUxpY2VuY2VBY2NlcHRhbmNlUmVxdWVzdBoyLnR1cmJvLnRyYWluaW5nLnYxLlJldm9rZUxpY2VuY2VBY2NlcHRhbmNlUmVzcG9uc2USfQoWTGlzdExpY2VuY2VBY2NlcHRhbmNlcxIwLnR1cmJvLnRyYWluaW5nLnYxLkxpc3RMaWNlbmNlQWNjZXB0YW5jZXNSZXF1ZXN0GjEudHVyYm8udHJhaW5pbmcudjEuTGlzdExpY2VuY2VBY2NlcHRhbmNlc1Jlc3BvbnNlEmsKEExpc3RNb2RlbFNvdXJjZXMSKi50dXJiby50cmFpbmluZy52MS5MaXN0TW9kZWxTb3VyY2VzUmVxdWVzdBorLnR1cmJvLnRyYWluaW5nLnYxLkxpc3RNb2RlbFNvdXJjZXNSZXNwb25zZRJ0ChNSZWdpc3Rlck1vZGVsU291cmNlEi0udHVyYm8udHJhaW5pbmcudjEuUmVnaXN0ZXJNb2RlbFNvdXJjZVJlcXVlc3QaLi50dXJiby50cmFpbmluZy52MS5SZWdpc3Rlck1vZGVsU291cmNlUmVzcG9uc2USbgoRUmVtb3ZlTW9kZWxTb3VyY2USKy50dXJiby50cmFpbmluZy52MS5SZW1vdmVNb2RlbFNvdXJjZVJlcXVlc3QaLC50dXJiby50cmFpbmluZy52MS5SZW1vdmVNb2RlbFNvdXJjZVJlc3BvbnNlEmsKEFN0YXJ0RHJpdmVTaWduSW4SKi50dXJiby50cmFpbmluZy52MS5TdGFydERyaXZlU2lnbkluUmVxdWVzdBorLnR1cmJvLnRyYWluaW5nLnYxLlN0YXJ0RHJpdmVTaWduSW5SZXNwb25zZRJuChFGaW5pc2hEcml2ZVNpZ25JbhIrLnR1cmJvLnRyYWluaW5nLnYxLkZpbmlzaERyaXZlU2lnbkluUmVxdWVzdBosLnR1cmJvLnRyYWluaW5nLnYxLkZpbmlzaERyaXZlU2lnbkluUmVzcG9uc2UynAUKFFR1cmJvVHJhaW5pbmdTZXJ2aWNlEl8KDExpc3RUZWFjaGVycxImLnR1cmJvLnRyYWluaW5nLnYxLkxpc3RUZWFjaGVyc1JlcXVlc3QaJy50dXJiby50cmFpbmluZy52MS5MaXN0VGVhY2hlcnNSZXNwb25zZRJrChBUcmFpblN0YXRpY01vZGVsEioudHVyYm8udHJhaW5pbmcudjEuVHJhaW5TdGF0aWNNb2RlbFJlcXVlc3QaKS50dXJiby50cmFpbmluZy52MS5UcmFpblN0YXRpY01vZGVsVXBkYXRlMAESawoQTGlzdFN0YXRpY01vZGVscxIqLnR1cmJvLnRyYWluaW5nLnYxLkxpc3RTdGF0aWNNb2RlbHNSZXF1ZXN0GisudHVyYm8udHJhaW5pbmcudjEuTGlzdFN0YXRpY01vZGVsc1Jlc3BvbnNlEm4KEURlbGV0ZVN0YXRpY01vZGVsEisudHVyYm8udHJhaW5pbmcudjEuRGVsZXRlU3RhdGljTW9kZWxSZXF1ZXN0GiwudHVyYm8udHJhaW5pbmcudjEuRGVsZXRlU3RhdGljTW9kZWxSZXNwb25zZRJnCg5UcmFpbkNvbXBvbmVudBIoLnR1cmJvLnRyYWluaW5nLnYxLlRyYWluQ29tcG9uZW50UmVxdWVzdBonLnR1cmJvLnRyYWluaW5nLnYxLlRyYWluQ29tcG9uZW50VXBkYXRlKAEwARJwChFTdHJlYW1pbmdUcmFpbmluZxIrLnR1cmJvLnRyYWluaW5nLnYxLlN0cmVhbWluZ1RyYWluaW5nUmVxdWVzdBoqLnR1cmJvLnRyYWluaW5nLnYxLlN0cmVhbWluZ1RyYWluaW5nVXBkYXRlKAEwAWIGcHJvdG8z");
var InstallStage = /* @__PURE__ */ ((InstallStage2) => {
  InstallStage2[InstallStage2["UNSPECIFIED"] = 0] = "UNSPECIFIED";
  InstallStage2[InstallStage2["ADMITTED"] = 1] = "ADMITTED";
  InstallStage2[InstallStage2["DOWNLOADING"] = 2] = "DOWNLOADING";
  InstallStage2[InstallStage2["DOWNLOADED"] = 3] = "DOWNLOADED";
  InstallStage2[InstallStage2["VERIFIED"] = 4] = "VERIFIED";
  return InstallStage2;
})(InstallStage || {});
var TrainStage = /* @__PURE__ */ ((TrainStage2) => {
  TrainStage2[TrainStage2["UNSPECIFIED"] = 0] = "UNSPECIFIED";
  TrainStage2[TrainStage2["ADMITTED"] = 1] = "ADMITTED";
  TrainStage2[TrainStage2["KEPT"] = 2] = "KEPT";
  TrainStage2[TrainStage2["EMBEDDING"] = 3] = "EMBEDDING";
  TrainStage2[TrainStage2["REDUCED"] = 4] = "REDUCED";
  TrainStage2[TrainStage2["MEASURED"] = 5] = "MEASURED";
  TrainStage2[TrainStage2["SEALED"] = 6] = "SEALED";
  TrainStage2[TrainStage2["LOADED"] = 7] = "LOADED";
  return TrainStage2;
})(TrainStage || {});
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
var demo = client(DemoService);
function describe(e) {
  return explain(e).text;
}

// demo/web/src/act.ts
var running = /* @__PURE__ */ new WeakSet();
var stoppable = /* @__PURE__ */ new WeakMap();
function hold(stop6, by) {
  const n2 = Math.max(0, (stoppable.get(stop6) ?? 0) + by);
  stoppable.set(stop6, n2);
  stop6.disabled = n2 === 0;
}
async function busy(button, job, stop6) {
  if (running.has(button)) return;
  running.add(button);
  const t0 = performance.now();
  const clock = el("span", { class: "elapsed", text: ` ${elapsed(0)}` });
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.append(clock);
  if (stop6) hold(stop6, 1);
  const tick = setInterval(() => clock.textContent = ` ${elapsed(performance.now() - t0)}`, 100);
  try {
    await job();
  } finally {
    clearInterval(tick);
    clock.remove();
    button.disabled = false;
    button.removeAttribute("aria-busy");
    running.delete(button);
    if (stop6) hold(stop6, -1);
  }
}
function submitter(ev) {
  const by = ev.submitter;
  if (by instanceof HTMLButtonElement) return by;
  return ev.target.querySelector("button[type=submit]");
}
function confirmed(label, question, act3, off2 = "") {
  const box5 = el("span", { class: "confirm" });
  const first = (why = "") => {
    const b = el("button", { type: "button", text: label, title: off2 || void 0 });
    b.disabled = off2 !== "";
    b.addEventListener("click", ask);
    box5.replaceChildren(b, why ? el("span", { class: "error", role: "alert", text: ` ${why}` }) : "");
    return b;
  };
  const ask = () => {
    const yes = el("button", { type: "button", class: "danger", text: "Yes" });
    const no = el("button", { type: "button", text: "Cancel" });
    no.addEventListener("click", () => first().focus());
    yes.addEventListener("click", () => {
      no.disabled = true;
      void busy(yes, async () => {
        try {
          await act3();
          if (box5.isConnected) first();
        } catch (e) {
          first(describe(e)).focus();
        }
      });
    });
    box5.replaceChildren(el("span", { class: "ask", text: question }), " ", yes, " ", no);
    no.focus();
  };
  box5.addEventListener("keydown", (ev) => {
    const no = box5.querySelector(".ask ~ button:not(.danger)");
    if (ev.key === "Escape" && no && !no.disabled) first().focus();
  });
  first();
  return box5;
}

// demo/web/src/doc.ts
var current = null;
var listeners = [];
function doc() {
  return current;
}
function setDoc(d) {
  current = d;
  for (const l of listeners) l(d);
}
function onDoc(l) {
  listeners.push(l);
}

// demo/web/src/marks.ts
function layerKey(layer, instance) {
  return `${layer}#${instance}`;
}
function layerName(layer, instance) {
  return instance ? `${layer} [${instance}]` : layer;
}
var NAMING = ["entities", "pii", "numbers", "language"];
function names(layer) {
  const l = layer.toLowerCase();
  return NAMING.some((n2) => l.includes(n2));
}
var ORDER = ["subwords", "tokens", "sentences", "chunks", "structure"];
function rank(a) {
  if (names(a.layer) && a.label) return 0;
  const at = ORDER.findIndex((n2) => a.layer.toLowerCase().includes(n2));
  return at < 0 ? ORDER.length + 1 : at + 1;
}
function colourOf(a) {
  if (names(a.layer) && a.label) return { key: `label:${a.label}`, name: a.label };
  return { key: `layer:${layerKey(a.layer, a.instance)}`, name: layerName(a.layer, a.instance) };
}
function kept(anns, filter2) {
  return filter2.size ? anns.filter((a) => filter2.has(layerKey(a.layer, a.instance))) : anns;
}
function painted(anns, filter2, lead) {
  let show3 = kept(anns, filter2);
  if (!filter2.size && lead !== "all") show3 = show3.filter((a) => lead === "naming" ? rank(a) === 0 : sentence2(a));
  return [...show3].sort((x, y) => rank(x) - rank(y));
}
function sentence2(a) {
  return a.layer.toLowerCase().includes("sentences");
}
function leadOf(segs) {
  if (segs.some((s2) => s2.annotations.some((a) => rank(a) === 0))) return "naming";
  return segs.some((s2) => s2.annotations.some(sentence2)) ? "sentences" : "all";
}
function legend(segs, filter2) {
  const any = leadOf(segs);
  const out = /* @__PURE__ */ new Map();
  const seen = /* @__PURE__ */ new Set();
  for (const s2 of segs) {
    for (const a of painted(s2.annotations, filter2, any)) {
      const c = colourOf(a);
      let sw = out.get(c.key);
      if (!sw) {
        sw = { key: c.key, name: c.name, index: out.size, items: 0 };
        out.set(c.key, sw);
      }
      const item = `${layerKey(a.layer, a.instance)}#${a.item}`;
      if (!seen.has(`${c.key} ${item}`)) {
        seen.add(`${c.key} ${item}`);
        sw.items++;
      }
    }
  }
  return [...out.values()];
}
function found(segs) {
  const items = /* @__PURE__ */ new Map();
  for (const s2 of segs) {
    for (const a of s2.annotations) {
      if (rank(a) !== 0) continue;
      const key3 = `${layerKey(a.layer, a.instance)}#${a.item}`;
      const it = items.get(key3);
      if (it) it.text += s2.text;
      else items.set(key3, { label: a.label, text: s2.text });
    }
  }
  const groups2 = /* @__PURE__ */ new Map();
  for (const it of items.values()) {
    const text = it.text.trim();
    const list5 = groups2.get(it.label) ?? [];
    if (text && !list5.includes(text)) list5.push(text);
    groups2.set(it.label, list5);
  }
  return [...groups2].map(([label, texts2]) => ({ label, texts: texts2 }));
}

// demo/web/src/tabs/annotated.ts
var COLOURS = 8;
function colour(index) {
  return `var(--m${index % COLOURS})`;
}
function tip(anns) {
  return anns.map((a) => `${a.layer}${a.instance ? ` (${a.instance})` : ""}${a.label ? `: ${a.label}` : ""}${a.score ? ` ${a.score.toFixed(3)}` : ""}`).join("\n");
}
function inspect(text, anns) {
  return [
    el("p", {}, el("span", { class: "quote", text })),
    table(["layer", "instance", "label", "score", "item"], anns.map((a) => [a.layer, a.instance, a.label, a.score ? a.score.toFixed(3) : "", String(a.item)]))
  ];
}
function annotated(segs, filter2) {
  const any = leadOf(segs);
  const swatches = legend(segs, filter2);
  const index = new Map(swatches.map((s2) => [s2.key, s2.index]));
  const panel = el("div", { class: "inspect", "aria-live": "polite" }, el("p", { class: "muted", text: "Pick a marked stretch of text to list what is annotated on it." }));
  const text = el("div", { class: "doc" });
  let picked2 = null;
  for (const s2 of segs) {
    const paint = painted(s2.annotations, filter2, any);
    if (!paint.length) {
      text.append(s2.text);
      continue;
    }
    const lead = paint[0];
    const second = paint.find((a) => colourOf(a).key !== colourOf(lead).key);
    const style = `--mark: ${colour(index.get(colourOf(lead).key) ?? 0)}${second ? `; --under: ${colour(index.get(colourOf(second).key) ?? 0)}` : ""}`;
    const cls = `${lead.item % 2 ? "alt" : ""}${second ? " under" : ""}`.trim();
    const mark = el("mark", { class: cls || void 0, style, title: tip(s2.annotations), tabindex: "0", role: "button", text: s2.text });
    const show3 = () => {
      picked2?.classList.remove("picked");
      picked2 = mark;
      mark.classList.add("picked");
      panel.replaceChildren(...inspect(s2.text, s2.annotations));
    };
    mark.addEventListener("click", show3);
    mark.addEventListener("focus", show3);
    text.append(mark);
  }
  const key3 = swatches.length ? el("div", { class: "legend" }, swatches.map((s2) => el("span", { class: "swatch", style: `--mark: ${colour(s2.index)}` }, el("i", {}), ` ${s2.name} `, el("span", { class: "muted", text: String(s2.items) })))) : el("p", { class: "muted", text: "Nothing is marked: pick a layer above to mark it in the text." });
  return el("div", { class: "annotated" }, key3, text, panel);
}
function foundList(segs) {
  const groups2 = found(segs);
  if (!groups2.length) return null;
  return el(
    "div",
    { class: "found" },
    el("h3", { text: "Found" }),
    el("dl", {}, groups2.map((g) => [el("dt", { text: `${g.label} (${g.texts.length})` }), el("dd", { text: g.texts.join(", ") })]))
  );
}

// demo/web/src/permit.ts
var writes = null;
var user = true;
var FOR_A_USER = "turbo.training.v1.TurboCatalogService";
var locked = [];
var changed = [];
function path(m) {
  return `/${m.parent.typeName}/${m.name}`;
}
function set(forwarded, hasUser = true) {
  const was = key();
  writes = forwarded ? new Set(forwarded) : null;
  user = forwarded === null || hasUser;
  for (const [box5, m] of locked) apply(box5, m);
  if (key() !== was) changed.forEach((l) => l());
}
function key() {
  return writes ? `${user} ${[...writes].sort().join(" ")}` : "none";
}
function onChange(l) {
  changed.push(l);
}
function off(m) {
  if (!writes) return "";
  if (!writes.has(path(m))) return `Off: the demo does not forward this write. Start it with --allow ${path(m).slice(1)}`;
  if (!user && m.parent.typeName === FOR_A_USER) return "Off: this acts for a user. Start the demo with --user NAME";
  return "";
}
function apply(box5, m) {
  const why = off(m);
  box5.inert = why !== "";
  box5.classList.toggle("off", why !== "");
  const id = `${box5.id}-off`;
  const old = document.getElementById(id);
  if (!why) return void old?.remove();
  const line = old ?? document.createElement("p");
  line.id = id;
  line.className = "muted off-why";
  line.textContent = why;
  if (!old) box5.after(line);
}
function lock(box5, m) {
  locked.push([box5, m]);
  apply(box5, m);
}

// demo/web/src/status.ts
function stale(was, now) {
  if (was === null) return true;
  if (was.up !== now.up) return true;
  return now.up && now.ready && !was.ready;
}

// demo/web/src/state.ts
var current2 = null;
var listeners2 = [];
var failures = [];
function capabilities() {
  if (!current2) {
    const mine = analysis.capabilities({});
    current2 = mine;
    mine.then(
      (c) => current2 === mine && listeners2.forEach((l) => l(c)),
      (e) => current2 === mine && failures.forEach((l) => l(e))
    );
  }
  return current2;
}
async function refresh() {
  current2 = null;
  for (; ; ) {
    const mine = capabilities();
    try {
      const c = await mine;
      if (current2 === mine) return c;
    } catch (e) {
      if (current2 === mine) throw e;
    }
  }
}
function onCapabilities(l) {
  listeners2.push(l);
}
function onCapabilitiesFailed(l) {
  failures.push(l);
}
function bundles(c, task) {
  return c.bundles.filter((b) => b.task === task).map((b) => b.name);
}
var last = null;
var watchers = [];
var readers = [];
function reachable() {
  return last === null ? null : last.up;
}
function setStatus(up, ready) {
  const was = last;
  last = { up, ready: up && ready };
  if (was === null || was.up !== up) watchers.forEach((l) => l(up, was === null ? null : was.up));
  if (stale(was, last)) readers.forEach((l) => l());
}
function onReachable(l) {
  watchers.push(l);
}
function onServed(l) {
  readers.push(l);
  if (last !== null) l();
}
function whyNone(what) {
  return reachable() === false ? "the server is not reachable" : `no ${what} is served`;
}
async function permits() {
  try {
    const d = await demo.describe({}, { signal: AbortSignal.timeout(3e3) });
    set(d.forwardedWrites, d.hasUser);
  } catch {
    set(null);
  }
}

// demo/web/src/latest.ts
function latest() {
  let n2 = 0;
  return () => {
    const mine = ++n2;
    return () => mine === n2;
  };
}

// demo/web/src/tabs/analyze.ts
var running2 = null;
var page = 0;
var filter = /* @__PURE__ */ new Set();
var shape = null;
var openSpanModels = /* @__PURE__ */ new Set();
var relationModels = /* @__PURE__ */ new Set();
var select = (id) => $(id);
function statusName(s2) {
  return LayerWireStatus[s2] ?? String(s2);
}
function layerRows(layers) {
  return table(
    ["layer", "instance", "status", "items", "annotator"],
    [...layers.values()].map((l) => [
      l.layer,
      l.instance,
      el("span", { class: l.status === 1 /* OK */ ? "" : "error", text: statusName(l.status), title: l.error ? LayerErrorKind[l.error.kind] : "" }),
      String(l.items),
      `${l.annotator} ${l.version}`.trim()
    ])
  );
}
function stop() {
  running2?.abort();
  running2 = null;
}
var settle = null;
async function run() {
  stop();
  const ac = new AbortController();
  running2 = ac;
  const note5 = $("analyze-note");
  const steps = $("analyze-steps");
  const layersBox = $("analyze-layers");
  steps.replaceChildren();
  layersBox.replaceChildren();
  $("analyze-window").replaceChildren();
  $("analyze-export-out").replaceChildren();
  setDoc(null);
  let graph = null;
  let views = [];
  let source = "";
  const layers = /* @__PURE__ */ new Map();
  const t0 = performance.now();
  let shown2 = false;
  settle = null;
  say(note5, "Analyzing\u2026", "busy");
  try {
    if (!document.getElementById("analyze-text").value.trim()) throw new Error("Enter a document first.");
    const model = select("analyze-model").value;
    const open2 = openSpanModels.has(model);
    const relationModel = relationModels.size ? select("analyze-relation-model").value : "";
    const labels = $("analyze-labels").value.split(",").map((label) => label.trim()).filter(Boolean);
    const relationLabels = $("analyze-relation-labels").value.split(",").map((label) => label.trim()).filter(Boolean);
    const threshold = $("analyze-threshold").valueAsNumber;
    const relationThreshold = $("analyze-relation-threshold").valueAsNumber;
    const guidelineText = $("analyze-guidelines").value.trim();
    const guidelineMap = /* @__PURE__ */ new Map();
    for (const line of guidelineText ? guidelineText.split("\n") : []) {
      const at = line.indexOf("=");
      if (at < 1 || at === line.length - 1) throw new Error("Use label=prompt on each guideline line.");
      const label = line.slice(0, at).trim();
      if (guidelineMap.has(label)) throw new Error(`Duplicate guideline for ${label}.`);
      guidelineMap.set(label, line.slice(at + 1).trim());
    }
    const prompts = guidelineText ? labels.map((label) => guidelineMap.get(label) ?? label) : [];
    if (open2 && labels.length === 0) throw new Error("Enter at least one open-span label.");
    if (open2 && !Number.isFinite(threshold)) throw new Error("Enter a threshold from 0 to 1.");
    if (relationModel && !open2) throw new Error("Select an open-span entity model before running relations.");
    if (relationModel && relationLabels.length === 0) throw new Error("Enter at least one relation label.");
    if (relationModel && !Number.isFinite(relationThreshold)) throw new Error("Enter a relation threshold from 0 to 1.");
    const pipelineSteps = open2 || relationModel ? ["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped"] : [];
    if (open2) pipelineSteps.push("turbo:gliner");
    if (relationModel) pipelineSteps.push("turbo:zero-shot-relations");
    const req = {
      docId: "page",
      text: document.getElementById("analyze-text").value,
      profile: select("analyze-profile").value,
      model,
      relationModel,
      offsets: 1 /* UTF16 */,
      steps: pipelineSteps,
      openSpan: open2 ? { labels, prompts, threshold, nested: $("analyze-nested").checked } : void 0,
      relations: relationModel ? {
        labels: relationLabels.map((name) => ({ name })),
        threshold: relationThreshold,
        withinSentence: $("analyze-relation-sentence").checked
      } : void 0
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
        if (!shown2) settle = reveal(layersBox.closest(".results"));
        shown2 = true;
      } else if (e.case === "stepFailed") {
        const f = e.value;
        steps.append(
          el("li", { class: "error", text: `${f.stepId} failed: ${f.failure?.message ?? ""}; skipped ${f.skippedDependents.join(", ") || "nothing"}` })
        );
      } else if (e.case === "complete") {
        graph = e.value.graph;
        views = e.value.views;
        source = e.value.sourceText;
        for (const l of e.value.layers) layers.set(`${l.layer}#${l.instance}`, l);
        layersBox.replaceChildren(layerRows(layers));
        steps.append(el("li", { class: "muted", text: `complete at ${ms} ms, ${graph.length} graph bytes` }));
      }
    }
    say(note5, "");
    if (graph) setDoc({ graph, views, source });
  } catch (e) {
    say(note5, explain(e));
  } finally {
    if (running2 === ac) running2 = null;
  }
}
function lanesView(w) {
  const box5 = el("div", { class: "lanes" });
  for (const lane of w.heatmapLanes) {
    const top = Math.max(1, ...lane.buckets.map((b) => Math.abs(b.score) || b.count));
    box5.append(
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
  return box5;
}
var freshWindow = latest();
async function showWindow() {
  const mine = freshWindow();
  const graph = doc()?.graph;
  const box5 = $("analyze-window");
  if (!graph) return void box5.replaceChildren();
  try {
    const picked2 = (shape ?? []).filter((l) => filter.has(layerKey(l.layer, l.instance))).map((l) => ({ layer: l.layer, instance: l.instance }));
    const base2 = { graph, view: 0, offsets: 1 /* UTF16 */, pageSize: 16e3, filterLayerInstances: picked2 };
    const [w, heat] = await Promise.all([
      analysis.inspectDocumentWindow({ ...base2, page }),
      analysis.inspectDocumentWindow({ ...base2, page: 0, heatmapBuckets: 96 })
    ]);
    if (!mine()) return;
    const win = w.window;
    shape = w.summary?.layers ?? [];
    const chips = shape.map((l) => {
      const key3 = layerKey(l.layer, l.instance);
      const on = filter.has(key3);
      const b = el("button", { type: "button", class: on ? "chip on" : "chip", "aria-pressed": String(on), text: `${layerName(l.layer, l.instance)} ${l.items}` });
      b.addEventListener("click", () => {
        if (on) filter.delete(key3);
        else filter.add(key3);
        void showWindow();
      });
      return b;
    });
    const list5 = foundList(w.segments);
    $("analyze-found").replaceChildren(...list5 ? [list5] : []);
    const nav = el("div", { class: "nav" });
    const prev = el("button", { type: "button", text: "\u2190 page" });
    const next = el("button", { type: "button", text: "page \u2192" });
    prev.toggleAttribute("disabled", !win || win.page === 0);
    next.toggleAttribute("disabled", !win || win.page + 1 >= win.pageCount);
    prev.addEventListener("click", () => (page = Math.max(0, page - 1), void showWindow()));
    next.addEventListener("click", () => (page += 1, void showWindow()));
    nav.append(prev, el("span", { class: "muted", text: win ? ` page ${win.page + 1} of ${win.pageCount}, units ${win.start}..${win.end} ` : "" }), next);
    const s2 = w.summary;
    box5.replaceChildren(
      el("h3", { text: "Document" }),
      s2 ? el("p", { class: "muted", text: `${s2.totalBytes} bytes, ${s2.totalUnits} UTF-16 units, ${s2.viewCount} views, ${s2.layerCount} layers. Pick layer instances to filter text annotations.` }) : "",
      el("div", { class: "chips" }, chips),
      nav,
      annotated(w.segments, filter),
      el("h3", { text: "Heatmap" }),
      heatKey("One row per layer, the document from start to end in 96 steps. Stronger colour is more items or a higher score in that step, relative to the row's greatest."),
      lanesView(heat)
    );
    settle?.();
    settle = null;
  } catch (e) {
    if (!mine()) return;
    box5.replaceChildren(failed(explain(e)));
  }
}
async function exportDoc() {
  const out = $("analyze-export-out");
  const graph = doc()?.graph;
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
    out.replaceChildren(failed(explain(e)));
  }
}
var formatsFailed = false;
var freshFormats = latest();
async function loadFormats() {
  const mine = freshFormats();
  try {
    const f = await analysis.listOutputFormats({});
    if (!mine()) return;
    options(select("analyze-format"), f.formats.map((x) => x.formatId), f.formats.map((x) => `${x.displayName} (${x.mediaType})`), "the server lists no format");
    if (formatsFailed) say($("analyze-note"), "");
    formatsFailed = false;
  } catch (e) {
    if (!mine()) return;
    const down = reachable() === false;
    if (!select("analyze-format").value) options(select("analyze-format"), [], void 0, down ? "the server is not reachable" : "the formats could not be listed");
    formatsFailed = !down;
    if (!down) say($("analyze-note"), explain(e));
  }
}
function groups() {
  $("analyze-open-span").hidden = !openSpanModels.has(select("analyze-model").value);
  $("analyze-relation-pick").hidden = relationModels.size === 0;
  $("analyze-relation").hidden = relationModels.size === 0 || select("analyze-relation-model").value === "";
}
function init2() {
  const go2 = $("analyze-run");
  go2.addEventListener("click", () => void busy(go2, run, $("analyze-stop")));
  $("analyze-stop").addEventListener("click", stop);
  const format = $("analyze-export");
  format.addEventListener("click", () => void busy(format, exportDoc));
  onDoc(() => {
    page = 0;
    filter = /* @__PURE__ */ new Set();
    shape = null;
    $("analyze-found").replaceChildren();
    void showWindow();
  });
  onCapabilities((c) => {
    openSpanModels = new Set(bundles(c, "TASK_OPEN_SPAN"));
    relationModels = new Set(bundles(c, "TASK_RELATION"));
    const models2 = [...bundles(c, "TASK_EMBED"), ...openSpanModels];
    options(select("analyze-profile"), ["", ...c.profiles.map((p) => p.name)], ["default", ...c.profiles.map((p) => `${p.name} v${p.version}`)]);
    options(select("analyze-model"), ["", ...models2], ["none", ...models2]);
    options(select("analyze-relation-model"), ["", ...relationModels], ["none", ...relationModels]);
    groups();
  });
  select("analyze-model").addEventListener("change", groups);
  select("analyze-relation-model").addEventListener("change", groups);
  let filled = false;
  onCapabilities(() => filled = true);
  onReachable((up) => {
    if (filled || up) return;
    for (const id of ["analyze-profile", "analyze-model", "analyze-relation-model", "analyze-format"]) options(select(id), [], void 0, "the server is not reachable");
  });
  onServed(() => void loadFormats());
}

// demo/web/src/tabs/collections.ts
var watching = null;
var note = () => $("coll-note");
var freshList = latest();
async function list() {
  const mine = freshList();
  const box5 = $("coll-list");
  try {
    const r = await search.listCollections({});
    if (!mine()) return;
    box5.replaceChildren(
      r.collections.length ? table(
        ["collection", "units", "mean length", "vocabulary", "alert at", "top terms", ""],
        r.collections.map((c) => [
          c.collectionName,
          String(c.totalUnits),
          c.meanLength.toFixed(1),
          String(c.vocabularySize),
          c.oovAlertThreshold.toFixed(2),
          c.topTerms.slice(0, 6).map((t) => `${t.term} ${t.documentFrequency}`).join(", "),
          remove(c.collectionName)
        ])
      ) : el("p", { class: "muted", text: "No collection yet: set one, or create an index with a collection name." })
    );
  } catch (e) {
    if (!mine()) return;
    box5.replaceChildren(el("p", { class: "error", text: reachable() === false ? "The server is not reachable, so its collections cannot be listed." : describe(e) }));
  }
}
function remove(name) {
  return confirmed("Delete", `Delete ${name}?`, async () => {
    await search.deleteCollection({ collectionName: name });
    await list();
  }, off(TurboSearchService.method.deleteCollection));
}
async function set2(ev) {
  ev.preventDefault();
  const name = input("coll-name").value.trim();
  if (!name) return say(note(), "Name the collection.", "error");
  await busy(submitter(ev), async () => {
    try {
      const r = await search.setCollection({
        collectionName: name,
        baselineTerms: input("coll-baseline").value.split(" ").map((t) => t.trim()).filter((t) => t.length),
        oovAlertThreshold: Math.min(1, Math.max(0, parseFloat(input("coll-threshold").value) || 0))
      });
      say(note(), `Set ${r.collection?.collectionName ?? name}.`);
      await list();
    } catch (e) {
      say(note(), explain(e));
    }
  });
}
function event(ev) {
  const r = ev.report;
  const at = (/* @__PURE__ */ new Date()).toLocaleTimeString();
  if (!r) return el("li", { text: `${at} ${ev.collectionName} from ${ev.sourceIndex}: no report` });
  return el(
    "li",
    {},
    el("span", { class: "badge", text: `OOV ${(r.oovRatio * 100).toFixed(1)}%` }),
    ` ${at} ${ev.collectionName} from ${ev.sourceIndex}: coverage ${(r.vocabularyCoverage * 100).toFixed(1)}%, ${r.totalUnits} units`,
    r.newTerms.length ? el("div", { class: "muted", text: `new: ${r.newTerms.slice(0, 24).join(" ")}${r.newTerms.length > 24 ? ` \u2026 ${r.newTerms.length} in all` : ""}` }) : ""
  );
}
async function watch() {
  const name = input("coll-name").value.trim();
  if (!name) return say(note(), "Name the collection to watch.", "error");
  const log = $("coll-events");
  await busy($("coll-watch"), async () => {
    stop2();
    const ac = new AbortController();
    watching = ac;
    say(note(), `Watching ${name}: index documents into an index of this collection to see its drift.`, "busy");
    try {
      for await (const ev of search.watchCollection({ collectionName: name, minOovRatio: Math.max(0, parseFloat(input("coll-min").value) || 0) }, { signal: ac.signal })) {
        log.prepend(event(ev));
        void list();
      }
      say(note(), "The server ended the watch.");
    } catch (e) {
      if (cancelled(e)) say(note(), "Stopped watching.");
      else say(note(), explain(e));
    } finally {
      if (watching === ac) watching = null;
    }
  }, $("coll-stop"));
}
function stop2() {
  watching?.abort();
  watching = null;
}
function init3() {
  $("coll-set").addEventListener("submit", set2);
  const setButton = $("coll-set").querySelector("button[type=submit]");
  const gate = () => {
    const why = off(TurboSearchService.method.setCollection);
    setButton.disabled = why !== "";
    setButton.title = why;
  };
  gate();
  onChange(() => {
    gate();
    void list();
  });
  const again = $("coll-refresh");
  again.addEventListener("click", () => void busy(again, list));
  onServed(() => void list());
  $("coll-watch").addEventListener("click", () => void watch());
  $("coll-stop").addEventListener("click", stop2);
}

// demo/web/src/brief.ts
var LISTED = 6;
var TEXT = 48;
function scalar(v) {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)));
  if (typeof v === "bigint") return v.toString();
  if (typeof v === "boolean") return v ? "true" : "";
  if (typeof v === "string") return v.length > TEXT ? `"${v.slice(0, TEXT - 1)}\u2026"` : `"${v}"`;
  if (v instanceof Uint8Array) return `${v.length} bytes`;
  return "";
}
function unset(v) {
  return v === void 0 || v === 0 || v === "" || v === false || v === 0n || v instanceof Uint8Array && v.length === 0;
}
function enumName(e, n2) {
  const v = e.value[n2];
  if (!v) return String(n2);
  const pre = e.sharedPrefix ?? "";
  return v.name.toLowerCase().startsWith(pre) ? v.name.slice(pre.length) : v.name;
}
function list2(f, v, depth) {
  if (f.listKind === "message") {
    const shown2 = v.slice(0, depth >= 2 ? 0 : 3).map((m) => `{${brief(f.message, m, depth + 1)}}`);
    return `[${v.length}]${shown2.length ? ` ${shown2.join(" ")}${v.length > shown2.length ? " \u2026" : ""}` : ""}`;
  }
  const one = (x) => f.listKind === "enum" ? enumName(f.enum, x) : scalar(x);
  return `[${v.slice(0, LISTED).map(one).join(", ")}${v.length > LISTED ? `, \u2026 ${v.length} in all` : ""}]`;
}
function field(f, v, depth) {
  switch (f.fieldKind) {
    case "scalar":
      return unset(v) && !f.proto.proto3Optional ? "" : scalar(v);
    case "enum":
      return v === void 0 ? "" : enumName(f.enum, v);
    case "message":
      if (v === void 0) return "";
      return depth >= 2 ? "{\u2026}" : `{${brief(f.message, v, depth + 1)}}`;
    case "list":
      return v.length ? list2(f, v, depth) : "";
    case "map": {
      const n2 = Object.keys(v).length;
      return n2 ? `{${n2} entries}` : "";
    }
  }
}
function brief(desc, m, depth = 0, skip = []) {
  const out = [];
  for (const member of desc.members) {
    if (skip.includes(member.localName)) continue;
    if (member.kind === "oneof") {
      const o = m[member.localName];
      const f = o?.case ? member.fields.find((x) => x.localName === o.case) : void 0;
      if (f) out.push(`${f.localName} ${field(f, o?.value, depth) || "{}"}`);
      continue;
    }
    const s2 = field(member, m[member.localName], depth);
    if (s2) out.push(`${member.localName} ${s2}`);
  }
  return out.join(", ");
}

// demo/web/src/tabs/xray.ts
var box = () => $("xray-out");
var from = () => $("xray-from");
var to = () => $("xray-to");
var unit = () => $("xray-offsets");
var UNITS = {
  utf8: [0 /* UTF8 */, "UTF-8 bytes"],
  utf16: [1 /* UTF16 */, "UTF-16 units"],
  cp: [2 /* CODE_POINT */, "code points"]
};
function viewName(instance, i) {
  return `${i}: ${instance || "source"}`;
}
function runs(x) {
  const text = el("div", { class: "doc xray" });
  for (const r of x.runs) {
    const where = `${r.parentStart}..${r.parentEnd} \u2192 ${r.childStart}..${r.childEnd}`;
    if (r.kind === 0 /* EQUAL */) text.append(r.childText);
    else if (r.kind === 2 /* INSERTED */) text.append(el("ins", { title: `inserted ${where}`, text: r.childText }));
    else if (r.kind === 3 /* DELETED */) text.append(el("del", { title: `deleted ${where}`, text: r.parentText }));
    else text.append(el("del", { title: `replaced ${where}`, text: r.parentText }), el("ins", { title: `replaced ${where}`, text: r.childText }));
  }
  return text;
}
var freshRender = latest();
async function render() {
  const mine = freshRender();
  const d = doc();
  if (!d) return void box().replaceChildren(el("p", { class: "muted", text: "Analyze a document first." }));
  if (d.views.length < 2) {
    return void box().replaceChildren(el("p", { class: "muted", text: "This document has one view: the profile ran no normalization step." }));
  }
  const a = parseInt(from().value, 10) || 0;
  const b = parseInt(to().value, 10) || 0;
  const [offsets, unitName] = UNITS[unit().value] ?? UNITS.utf16;
  try {
    const r = await analysis.inspectDocumentWindow({ graph: d.graph, view: a, offsets, pageSize: 1, page: 0, xrayTargetView: b });
    if (!mine()) return;
    const x = r.xray;
    if (!x) return void box().replaceChildren(el("p", { class: "muted", text: "The server gave no X-Ray for these views." }));
    const edits2 = x.runs.filter((r2) => r2.kind !== 0 /* EQUAL */).length;
    box().replaceChildren(
      el("p", { class: "muted", text: `${edits2} edits from view ${x.fromView} to view ${x.toView}, offsets in ${unitName}. Hover an edit for its spans.` }),
      runs(x),
      el("h3", { text: "Steps" }),
      table(
        ["view", "parent", "instance", "annotator", "edits", "bytes in", "bytes out"],
        x.steps.map((s2) => [String(s2.view), s2.parentView === void 0 ? "" : String(s2.parentView), s2.instance, `${s2.annotator} ${s2.version}`, String(s2.editCount), String(s2.parentBytes), String(s2.childBytes)])
      )
    );
  } catch (e) {
    if (!mine()) return;
    box().replaceChildren(failed(explain(e)));
  }
}
function init4() {
  const again = lazy($("analyze-xray"), () => void render());
  onDoc((d) => {
    const views = d?.views ?? [];
    const values = views.map((_, i) => String(i));
    const labels = views.map((v, i) => viewName(v.instance, i));
    options(from(), values, labels, "analyze a document first");
    options(to(), values, labels, "analyze a document first");
    from().value = "0";
    to().value = String(Math.max(0, views.length - 1));
    again();
  });
  from().addEventListener("change", () => void render());
  unit().addEventListener("change", () => void render());
  to().addEventListener("change", () => void render());
}

// demo/web/src/tabs/drawer.ts
var PAGE = 200;
var result = null;
var picked = "";
var shown = PAGE;
var box2 = () => $("drawer-out");
var viewSelect = () => $("drawer-view");
function key2(l) {
  return `${l.id}#${l.instance}`;
}
function kindOf(name) {
  return TypedAnnotationSchema.fields.find((f) => f.localName === name)?.message;
}
function rows(r, l) {
  const list5 = l.annotations.slice(0, shown).map((a, i) => {
    const spans = a.spans.map((s2) => `${s2.start}..${s2.end}`).join(" ");
    const text = a.spans.map((s2) => r.rawText.slice(s2.start, s2.end)).join(" \u2026 ");
    const kind = kindOf(a.value.case);
    const detail = kind && a.value.value ? brief(kind, a.value.value) : "";
    return [String(i), spans, el("span", { class: "quote", text }), a.value.case ?? "", a.probability === void 0 ? "" : a.probability.toFixed(3), el("span", { class: "vec", text: detail })];
  });
  const more = el("button", { type: "button", text: `Show ${Math.min(PAGE, l.annotations.length - shown)} more` });
  more.addEventListener("click", () => (shown += PAGE, draw()));
  return el(
    "div",
    {},
    el("p", { class: "muted", text: `${l.annotations.length} annotations on ${l.scope === 0 ? "text" : l.scope === 1 ? "an aligned layer" : "the document"}${l.anchorLayer ? `, anchored to ${l.anchorLayer}` : ""}${l.modelRefs.length ? `, models ${l.modelRefs.join(", ")}` : ""}.` }),
    table(["#", "span", "text", "kind", "p", "fields"], list5, "drawer"),
    l.annotations.length > shown ? more : ""
  );
}
function draw() {
  const r = result;
  if (!r) return;
  const chips = r.layers.map((l) => {
    const bad = l.status !== 1 /* OK */;
    const b = el("button", {
      type: "button",
      class: `chip${key2(l) === picked ? " on" : ""}${bad ? " bad" : ""}`,
      title: bad ? `${LayerWireStatus[l.status]}: ${l.error ? LayerErrorKind[l.error.kind] : ""}` : "",
      text: `${l.id}${l.instance ? `#${l.instance}` : ""} ${l.annotations.length}`
    });
    b.addEventListener("click", () => {
      picked = key2(l);
      shown = PAGE;
      draw();
    });
    return b;
  });
  const layer = r.layers.find((l) => key2(l) === picked);
  box2().replaceChildren(
    el("p", { class: "muted", text: `${r.layers.length} layers on view ${r.view}. Pick one to open its annotations.` }),
    el("div", { class: "chips" }, chips),
    layer ? rows(r, layer) : "",
    r.models.length ? el("h3", { text: "Models" }) : "",
    r.models.length ? table(
      ["ref", "annotator", "version", "bundle", "ran on"],
      r.models.map((m) => [m.modelRef, m.annotator, m.version, m.bundle ? `${m.bundle.id}@${m.bundle.revision}` : "", m.execution ? `${m.execution.backend} ${m.execution.device}`.trim() : ""])
    ) : ""
  );
}
var freshRender2 = latest();
async function render2() {
  const mine = freshRender2();
  const d = doc();
  if (!d) return void box2().replaceChildren(el("p", { class: "muted", text: "Analyze a document first." }));
  try {
    const r = await analysis.inspectDocumentGraph({ docId: "page", graph: d.graph, view: parseInt(viewSelect().value, 10) || 0, offsets: 1 /* UTF16 */ });
    if (!mine()) return;
    result = r;
    if (!result.layers.some((l) => key2(l) === picked)) {
      const first = result.layers.find((l) => l.annotations.length);
      picked = first ? key2(first) : "";
    }
    shown = PAGE;
    draw();
  } catch (e) {
    if (!mine()) return;
    result = null;
    box2().replaceChildren(failed(explain(e)));
  }
}
function init5() {
  const again = lazy($("analyze-drawer"), () => void render2());
  onDoc((d) => {
    const views = d?.views ?? [];
    options(viewSelect(), views.map((_, i) => String(i)), views.map((v, i) => viewName(v.instance, i)), "analyze a document first");
    viewSelect().value = "0";
    again();
  });
  viewSelect().addEventListener("change", () => void render2());
}

// demo/web/src/csv.ts
function inert(v) {
  const lead = v.trimStart().charAt(0);
  if (lead !== "=" && lead !== "+" && lead !== "-" && lead !== "@") return v;
  const n2 = v.trim();
  return n2 !== "" && Number.isFinite(Number(n2)) ? v : `'${v}`;
}
function field2(raw) {
  const v = inert(raw);
  const quote = v.includes(",") || v.includes('"') || v.includes("\n") || v.includes("\r") || v.startsWith(" ") || v.endsWith(" ");
  return quote ? `"${v.replaceAll('"', '""')}"` : v;
}
function csv(rows2) {
  return rows2.map((r) => r.map(field2).join(",") + "\r\n").join("");
}
function tsv(rows2) {
  const flat = (v) => v.replaceAll("	", " ").replaceAll("\r\n", " ").replaceAll("\n", " ").replaceAll("\r", " ");
  return rows2.map((r) => r.map((v) => inert(flat(v))).join("	")).join("\n");
}
function fileName(name) {
  let out = "";
  for (const ch of name.normalize("NFC")) {
    const keep = ch.toLowerCase() !== ch.toUpperCase() || ch >= "0" && ch <= "9" || ch > "\x7F";
    out += keep ? ch : out.endsWith("-") || !out ? "" : "-";
  }
  while (out.endsWith("-")) out = out.slice(0, -1);
  return `${out || "result"}.csv`;
}

// demo/web/src/tabs/take.ts
async function copy(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = el("textarea", { class: "offscreen", "aria-hidden": "true" });
    area.value = text;
    document.body.append(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}
function take(name, rows2) {
  const copyButton = el("button", { type: "button", class: "small", text: "Copy", title: "Copy as tab-separated rows, which a spreadsheet pastes into cells" });
  const said = el("span", { class: "muted", role: "status" });
  copyButton.addEventListener("click", async () => {
    const all = rows2();
    said.textContent = await copy(tsv(all)) ? `Copied ${all.length - 1} rows.` : "The browser did not allow the copy.";
    setTimeout(() => said.textContent = "", 4e3);
  });
  const save = el("button", { type: "button", class: "small", text: "Download CSV" });
  save.addEventListener("click", () => download(fileName(name), [csv(rows2())], "text/csv;charset=utf-8"));
  return el("div", { class: "take", "data-take": name }, copyButton, save, said);
}

// demo/web/src/heat.ts
function range(m) {
  let lo = Infinity;
  let hi = -Infinity;
  m.forEach(
    (row, i) => row.forEach((v, j) => {
      if (i === j || !Number.isFinite(v)) return;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    })
  );
  return lo <= hi ? [lo, hi] : [0, 1];
}
function place(v, lo, hi) {
  if (!(hi > lo)) return 1;
  return Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
}
function best(m, i) {
  let at = -1;
  m[i].forEach((v, j) => {
    if (j !== i && Number.isFinite(v) && (at < 0 || v > m[i][at])) at = j;
  });
  return at;
}
function strength(t) {
  return 0.06 + 0.8 * t;
}

// demo/web/src/tabs/simtable.ts
var own = true;
function shade(t) {
  return `background: rgba(var(--heat), ${strength(t).toFixed(3)})`;
}
function simTable(labels, sim, name = "Cosine similarity") {
  const [lo, hi] = range(sim);
  const cells = sim.map((row, i) => {
    const top = best(sim, i);
    return row.map((x, j) => el("td", { class: j === top ? "cell best" : "cell", title: j === top ? "this row's best match" : void 0, text: x.toFixed(3) }));
  });
  const paint = () => {
    sim.forEach((row, i) => row.forEach((x, j) => cells[i][j].style.cssText = shade(own ? place(x, lo, hi) : place(x, 0, 1))));
  };
  const pick = el("input", { type: "checkbox" });
  pick.checked = own;
  pick.addEventListener("change", () => {
    own = pick.checked;
    paint();
  });
  paint();
  const table2 = el(
    "table",
    { class: "sim" },
    el("tr", {}, el("th", {}), labels.map((_, j) => el("th", { text: `#${j + 1}` }))),
    labels.map((l, i) => el("tr", {}, el("td", { class: "text", title: l.full, text: `#${i + 1} ${l.short}` }), cells[i]))
  );
  const scale2 = el(
    "div",
    { class: "scale" },
    el("label", {}, pick, ` colour by this matrix's own range, ${lo.toFixed(3)} to ${hi.toFixed(3)}`),
    el("span", { class: "muted", text: "outlined: each row's best match" })
  );
  const rows2 = () => [["", ...labels.map((l) => l.full)], ...sim.map((row, i) => [labels[i].full, ...row.map((x) => x.toFixed(6))])];
  return el("div", {}, scale2, table2, take(name, rows2));
}

// demo/web/src/tabs/embed.ts
var models = /* @__PURE__ */ new Map();
var listed = [];
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
var PICKED = "embed-models";
function renderModels() {
  const box5 = $("embed-models");
  box5.replaceChildren();
  if (models.size === 0) {
    const down = reachable() === false;
    box5.append(el("p", { class: down ? "error empty" : "muted empty", text: down ? "The server is not reachable, so its embedding models cannot be listed." : "No embedding model is served: add a name below." }));
  }
  for (const [name, m] of models) {
    const cb = el("input", { type: "checkbox" });
    cb.checked = m.checked;
    cb.disabled = !m.meta;
    cb.addEventListener("change", () => {
      m.checked = cb.checked;
      remember(PICKED, [...models].filter(([, x]) => x.checked).map(([n2]) => n2).join("\n"));
    });
    box5.append(
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
async function addModels(names2) {
  const fresh = names2.filter((n2) => !models.get(n2)?.meta);
  const kept2 = recallList(PICKED);
  for (const n2 of fresh) models.set(n2, { meta: null, error: "", checked: kept2 ? kept2.includes(n2) : true });
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
  return Math.sqrt(a.reduce((s2, x) => s2 + x * x, 0));
}
function matrix(vectors) {
  return vectors.map((a) => vectors.map((b) => cosine(a, b)));
}
function half(bits) {
  const s2 = bits & 32768 ? -1 : 1;
  const e = bits >> 10 & 31;
  const f = bits & 1023;
  if (e === 0) return s2 * f * 2 ** -24;
  if (e === 31) return f ? NaN : s2 * Infinity;
  return s2 * (1 + f / 1024) * 2 ** (e - 15);
}
function vectorsOf(reply) {
  const out = reply.outputs[0];
  const raw = reply.rawOutputContents[0];
  if (!out || !raw) throw new Error("the reply has no vectors");
  const [rows2, dim] = out.shape.map(Number);
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const width = out.datatype === "FP32" ? 4 : out.datatype === "FP16" || out.datatype === "BF16" ? 2 : 0;
  if (!width) throw new Error(`vectors in ${out.datatype} are not shown`);
  if (raw.byteLength !== rows2 * dim * width) throw new Error(`${raw.byteLength} bytes for a ${rows2}x${dim} ${out.datatype} tensor`);
  const v = [];
  for (let r = 0; r < rows2; r++) {
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
  const role2 = document.getElementById("embed-role").value;
  if (role2) p.prompt_role = param({ case: "stringParam", value: role2 });
  const max = parseInt(input("embed-max").value, 10);
  if (max > 0) {
    p.max_tokens = param({ case: "int64Param", value: BigInt(max) });
    p.truncate = param({ case: "stringParam", value: "TRUNCATE_RIGHT" });
  }
  return p;
}
async function run2() {
  const texts2 = document.getElementById("embed-texts").value.split("\n").map((t) => t.trim()).filter((t) => t.length);
  const chosen = [...models].filter(([, m]) => m.checked && m.meta).map(([n2]) => n2);
  const err = $("embed-error");
  say(err, "");
  if (!texts2.length || !chosen.length) {
    say(err, !texts2.length ? "Enter at least one text." : "Pick at least one model.", "error");
    return;
  }
  const encoder = new TextEncoder();
  const results = [];
  for (const name of chosen) {
    const t0 = performance.now();
    const meta = models.get(name).meta;
    try {
      const reply = await inference.modelInfer({
        modelName: name,
        parameters: parameters(),
        inputs: [{ name: "texts", datatype: "BYTES", shape: [BigInt(texts2.length)], contents: { bytesContents: texts2.map((t) => encoder.encode(t)) } }]
      });
      results.push({ name, meta, ms: performance.now() - t0, reply, vectors: vectorsOf(reply) });
    } catch (e) {
      results.push({ name, meta, ms: performance.now() - t0, error: describe(e) });
    }
  }
  render3(texts2, results);
}
function short(t, n2 = 28) {
  const chars = typeof Intl !== "undefined" && Intl.Segmenter ? Array.from(new Intl.Segmenter(void 0, { granularity: "grapheme" }).segment(t), (g) => g.segment) : Array.from(t);
  return chars.length > n2 ? chars.slice(0, n2 - 1).join("") + "\u2026" : t;
}
function card(texts2, r) {
  const head = el("div", { class: "head" }, el("h3", {}, r.name, el("span", { class: "tag", text: tier(r.meta) })));
  if (!r.vectors || !r.reply) {
    return el("div", { class: "card" }, head, el("p", { class: "error", text: r.error ?? "" }), stat(`${r.ms.toFixed(1)} ms`, "browser round trip"));
  }
  const v = r.vectors;
  const p = r.reply.parameters;
  const str2 = (k) => p[k]?.parameterChoice.case === "stringParam" ? String(p[k].parameterChoice.value) : "";
  const sim = simTable(texts2.map((t) => ({ short: short(t, 22), full: t })), matrix(v), `Cosine similarity ${r.name}`);
  const whole = () => [["text", "norm", ...v[0].map((_, d) => `d${d}`)], ...v.map((row, i) => [texts2[i], String(norm(row)), ...row.map(String)])];
  const previews = el(
    "table",
    {},
    el("tr", {}, el("th", { class: "text", text: "text" }), el("th", { text: "norm" }), el("th", { class: "text", text: "first values" })),
    v.map(
      (row, i) => el(
        "tr",
        {},
        el("td", { class: "text", title: texts2[i], text: `#${i + 1} ${short(texts2[i], 18)}` }),
        el("td", { text: norm(row).toFixed(4) }),
        el("td", {
          class: "vec",
          text: row.slice(0, 6).map((x) => (x < 0 ? "" : " ") + x.toFixed(4)).join(" ") + (row.length > 6 ? "  \u2026" : "")
        })
      )
    )
  );
  const device = [str2("backend"), str2("arch")].filter(Boolean).join(" ");
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
      str2("compute_dtype") ? stat(after(str2("compute_dtype"), "DTYPE_"), "computed in") : null
    ),
    el("h2", { text: "Cosine similarity" }),
    sim,
    el("details", {}, el("summary", { text: "Vectors" }), previews, take(`Vectors ${r.name}`, whole))
  );
}
function comparison(texts2, ok) {
  const sims = ok.map((r) => matrix(r.vectors));
  const all = sims.map((m) => range(m));
  const lo = Math.min(...all.map((r) => r[0]));
  const hi = Math.max(...all.map((r) => r[1]));
  const rows2 = [];
  let widest = 0;
  for (let i = 0; i < texts2.length; i++) {
    for (let j = i + 1; j < texts2.length; j++) {
      const vals = sims.map((m) => m[i][j]);
      const spread2 = Math.max(...vals) - Math.min(...vals);
      widest = Math.max(widest, spread2);
      rows2.push(
        el(
          "tr",
          {},
          el("td", { class: "text", title: `${texts2[i]}
${texts2[j]}`, text: `#${i + 1} \xB7 #${j + 1}` }),
          vals.map((x) => el("td", { class: "cell", style: shade(place(x, lo, hi)), text: x.toFixed(4) })),
          el("td", { text: spread2.toFixed(4) })
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
  const pairs = () => {
    const out = [["text a", "text b", ...ok.map((r) => `${r.name} (${tier(r.meta)})`), "spread"]];
    for (let i = 0; i < texts2.length; i++) {
      for (let j = i + 1; j < texts2.length; j++) {
        const vals = sims.map((m) => m[i][j]);
        out.push([texts2[i], texts2[j], ...vals.map((x) => x.toFixed(6)), (Math.max(...vals) - Math.min(...vals)).toFixed(6)]);
      }
    }
    return out;
  };
  const tableEl = el(
    "table",
    { class: "sim cmp" },
    el("tr", {}, el("th", { class: "text", text: "pair" }), ok.map((r) => el("th", { text: `${r.name} (${tier(r.meta)})` })), el("th", { text: "spread" })),
    rows2,
    el("tr", {}, el("th", { class: "text wrap", text: `lowest cosine to ${ok[0].name}, same text` }), agree.map((a) => el("td", { text: a })), el("td", {}))
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
    take("Side by side", pairs),
    widest === 0 ? el("p", { class: "note", text: "Every model gave the same numbers: on this device these tiers compute alike. A device that runs FASTEST in a narrower type shows the difference here." }) : null
  );
}
function render3(texts2, results) {
  const out = $("embed-results");
  out.replaceChildren();
  const ok = results.filter((r) => r.vectors);
  if (ok.length > 1 && texts2.length > 1) out.append(comparison(texts2, ok));
  out.append(el("div", { class: "cards" }, results.map((r) => card(texts2, r))));
  reveal(out);
}
function init6() {
  $("embed-add").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const name = input("embed-add-name").value.trim();
    if (!name) return;
    void busy(submitter(ev), async () => {
      await addModels([name]);
      input("embed-add-name").value = "";
    });
  });
  onReset("embed", () => {
    forgetOwn(PICKED);
    for (const m of models.values()) m.checked = m.meta !== null;
    renderModels();
  });
  const go2 = $("embed-run");
  go2.addEventListener("click", () => void busy(go2, run2));
  onCapabilities((c) => {
    say($("embed-error"), "");
    const served = bundles(c, "TASK_EMBED");
    for (const n2 of listed) if (!served.includes(n2)) models.delete(n2);
    listed = served;
    void addModels(served);
  });
  onReachable(renderModels);
  onCapabilitiesFailed((e) => {
    renderModels();
    if (reachable() === false) say($("embed-error"), "");
    else say($("embed-error"), explain(e));
  });
}

// demo/web/src/about.ts
var WHERE = "Load a bundle on the Server tab, or install a model on the Models tab.";
function needs(tab, tasks) {
  const has = (t) => tasks.includes(t);
  if (tab === "embed" && !has("TASK_EMBED")) return `Needs: an embedding model. The server serves none. ${WHERE}`;
  if (tab === "semantic" && !has("TASK_EMBED") && !has("TASK_RERANK") && !has("TASK_EXTRACTIVE_QA")) {
    return `Needs: an embedding model, a reranker or a QA model. The server serves none. ${WHERE}`;
  }
  if (tab === "search" && !has("TASK_EMBED")) return `Needs: an embedding model to create an index. The server serves none. ${WHERE}`;
  return "";
}
var HINTED = ["embed", "semantic", "search"];

// demo/web/src/tabs/hints.ts
function init7() {
  const show3 = (tasks) => {
    for (const tab of HINTED) {
      const line = $(`needs-${tab}`);
      const text = tasks ? needs(tab, tasks) : "";
      line.textContent = text;
      line.hidden = text === "";
    }
  };
  onCapabilities((c) => show3(c.bundles.map((b) => b.task)));
  onReachable((up) => {
    if (!up) show3(null);
  });
}

// demo/web/src/catalogfilter.ts
var PURPOSES = {
  COMPONENT_DENSE_ENCODER: "Embeds text for semantic search",
  COMPONENT_SPARSE_ENCODER: "Sparse embeddings for term-weighted search",
  COMPONENT_LATE_INTERACTION_ENCODER: "Per-token embeddings for late-interaction search",
  COMPONENT_ENTITY_EXTRACTOR: "Finds named entities",
  COMPONENT_SCHEMA_EXTRACTOR: "Extracts the fields of a schema",
  COMPONENT_RELATION_EXTRACTOR: "Finds relations between entities",
  COMPONENT_ZERO_SHOT_CATEGORIZER: "Classifies text into labels given at request time",
  COMPONENT_EXTRACTIVE_QA: "Answers questions from a passage",
  COMPONENT_SENTENCE_DETECTOR: "Splits text into sentences",
  COMPONENT_ASPECT_SENTIMENT: "Sentiment by aspect",
  COMPONENT_DOCUMENT_CATEGORIZER: "Classifies documents",
  COMPONENT_POS: "Tags parts of speech",
  COMPONENT_COREFERENCE_RESOLVER: "Links mentions of the same thing"
};
var KIND_PURPOSES = {
  KIND_DATASET: "Data to train and evaluate with",
  KIND_LEXICON: "Word list",
  KIND_MODEL: "Model files",
  KIND_HEAD: "Head on a shared encoder"
};
function purpose(e) {
  const c = e.component ?? "";
  if (c) return PURPOSES[c] ?? sentence3(plain2(c, "COMPONENT_"));
  return KIND_PURPOSES[e.kind] ?? sentence3(kindName(e.kind));
}
function sentence3(text) {
  const first = [...text][0] ?? "";
  return first.toUpperCase() + text.slice(first.length);
}
function kindName(kind) {
  const name = kind.startsWith("KIND_") ? kind.slice(5) : kind;
  return name.toLowerCase().split("_").join(" ");
}
function plain2(value, prefix2) {
  const name = value.startsWith(prefix2) ? value.slice(prefix2.length) : value;
  return name.toLowerCase().split("_").join(" ");
}
function fold(text) {
  const cased = text.toLowerCase().toUpperCase().toLowerCase();
  return cased.split("i\u0307").join("i").split("\u03C2").join("\u03C3");
}
function space(ch) {
  return ch.trim() === "" || ch === "\x85";
}
function words(text) {
  const out = [];
  let word = "";
  for (const ch of text) {
    if (!space(ch)) {
      word += ch;
      continue;
    }
    if (word) out.push(fold(word));
    word = "";
  }
  if (word) out.push(fold(word));
  return out;
}
function matches(e, kind, text) {
  if (kind && e.kind !== kind) return false;
  const hay = fold(`${e.id} ${kindName(e.kind)} ${e.kind} ${purpose(e)} ${e.language} ${e.licence}`);
  return words(text).every((w) => hay.includes(w));
}
function kinds(entries) {
  const count2 = /* @__PURE__ */ new Map();
  for (const e of entries) count2.set(e.kind, (count2.get(e.kind) ?? 0) + 1);
  return [...count2];
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
      const take2 = Math.min(64 - this.filled, data.length);
      this.block.set(data.subarray(0, take2), this.filled);
      this.filled += take2;
      i = take2;
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
  const s2 = new Sha256();
  const reader = b.stream().getReader();
  let read = 0;
  for (; ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    s2.update(value);
    read += value.length;
    progress?.(read);
  }
  return s2.hex();
}

// demo/web/src/tabs/licences.ts
function who(r) {
  if (!r.user) return { user: "", canChange: false, why: "read-only: the demo was started without --user" };
  if (!r.admin) return { user: r.user, canChange: false, why: "read-only: the server was started without --admin or --zoo" };
  return { user: r.user, canChange: true, why: "" };
}
var NOTICES = {
  CLASS_SHARE_ALIKE: "Share-alike: you may use it, and what you share that is built from it carries the same licence. This project does not redistribute it.",
  CLASS_LGPL: "LGPL: you may use it; changes to it that you distribute are released under the LGPL.",
  CLASS_GPL: "GPL: software you distribute that includes it is released under the GPL.",
  CLASS_LOCAL_ONLY: "Local only: restricted or non-commercial terms. It is fetched only from its publisher, stays on this machine, and no model trained on it is shipped. Not for commercial use."
};
function textSha256(text) {
  return new Sha256().update(new TextEncoder().encode(text)).hex();
}
function address(text) {
  let secure = false;
  try {
    secure = new URL(text).protocol === "https:";
  } catch {
    secure = false;
  }
  if (!secure) return el("code", { text });
  return el("a", { href: text, target: "_blank", rel: "noopener noreferrer", text });
}
function termsBlock(e, t, w, accepted) {
  const sources = e.trainedOn.map((s2) => `${s2.name} (${s2.licence})`).join(", ");
  const facts2 = el(
    "dl",
    {},
    el("dt", { text: "Licence" }),
    el("dd", { text: `${t.licence}, ${e.licenceClass}` }),
    el("dt", { text: "Terms" }),
    el("dd", {}, address(t.text)),
    e.attribution ? [el("dt", { text: "Attribution" }), el("dd", { text: e.attribution })] : null,
    sources ? [el("dt", { text: "Trained on" }), el("dd", { text: sources })] : null
  );
  const notice = el("p", { class: "muted", text: NOTICES[e.licenceClass] ?? "" });
  const body2 = [facts2, notice];
  const refused = off(TurboCatalogService.method.recordLicenceAcceptance);
  if (w.canChange && refused) {
    body2.push(el("p", { class: "muted", text: refused }));
  } else if (w.canChange) {
    const read = el("input", { type: "checkbox" });
    const button = el("button", { type: "button", text: `Accept ${t.licence}` });
    button.disabled = true;
    read.addEventListener("change", () => button.disabled = !read.checked);
    button.addEventListener("click", () => void busy(button, () => accepted(t)));
    body2.push(el("label", {}, read, ` I have read the ${t.licence} terms above and accept them`), " ", button);
  } else {
    body2.push(el("p", { class: "muted", text: w.why }));
  }
  return el("details", {}, el("summary", { text: `${t.licence} terms to accept` }), body2);
}
async function accept(e, t) {
  const r = await catalog.recordLicenceAcceptance({
    licence: t.licence,
    terms: t.licence.startsWith("LicenseRef-") ? t.text : "",
    artifactId: e.id,
    textSha256: textSha256(t.text)
  });
  return `${r.user} accepted ${r.accepted?.licence ?? t.licence} at ${when(r.acceptedAt)}.`;
}
function when(seconds) {
  return new Date(Number(seconds) * 1e3).toLocaleString();
}
function acceptedTable(r, w, revoked) {
  if (!w.user) return el("p", { class: "muted", text: w.why });
  if (!r.accepted.length) return el("p", { class: "muted", text: `${w.user} has accepted no terms.` });
  return table(
    ["licence", "terms", "accepted", "on the page of", ""],
    r.accepted.map((a) => {
      const t = a.terms;
      let control = "";
      if (t && w.canChange) {
        control = confirmed("Revoke", `Revoke ${t.licence}?`, () => revoked(t), off(TurboCatalogService.method.revokeLicenceAcceptance));
      }
      return [t?.licence ?? "", t ? address(t.text) : "", when(a.acceptedAt), a.artifactId, control];
    })
  );
}
async function revoke(t) {
  const r = await catalog.revokeLicenceAcceptance({ licence: t.licence, terms: t.licence.startsWith("LicenseRef-") ? t.text : "" });
  return r.wasAccepted ? `${r.user} revoked ${t.licence}.` : `${r.user} had not accepted ${t.licence}.`;
}

// demo/web/src/modelfile.ts
function nc(licence) {
  return /^CC-BY-NC(-|$)/.test(licence);
}
function nonCommercial(e) {
  return nc(e.licence) || e.trainedOn.some((s2) => nc(s2.licence));
}
function mismatch(f, got, sha256) {
  if (BigInt(got) !== f.size) return `${f.path}: ${got} bytes arrived, the catalog lists ${f.size}`;
  if (sha256 !== f.sha256) return `${f.path}: the bytes have SHA-256 ${sha256}, the catalog lists ${f.sha256}`;
  return "";
}

// demo/web/src/tabs/download.ts
async function download2(e, f, status3, signal) {
  const parts = [];
  const hash = new Sha256();
  let got = 0;
  for await (const m of catalog.readModelFile({ artifactId: e.id, path: f.path, sha256: f.sha256 }, { signal })) {
    if (m.frame.case !== "data") continue;
    const bytes2 = new Uint8Array(m.frame.value);
    parts.push(bytes2);
    hash.update(bytes2);
    got += bytes2.length;
    status3.textContent = `${e.id}/${f.path}: ${size(got)} of ${size(f.size)}`;
  }
  const wrong = mismatch(f, got, hash.hex());
  if (wrong) throw new Error(wrong);
  const url = URL.createObjectURL(new Blob(parts, { type: "application/octet-stream" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = f.path.split("/").pop() ?? f.path;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 6e4);
  status3.textContent = `${e.id}/${f.path}: ${size(f.size)}, SHA-256 checked, saved.`;
}

// demo/web/src/sourcetext.ts
function signInText(s2) {
  if (s2 === 2 /* GOOGLE */) return "Google sign-in";
  if (s2 === 3 /* API_KEY */) return "API key";
  if (s2 === 4 /* S3_KEYS */) return "S3 keys";
  if (s2 === 1 /* ENVIRONMENT */) return "the server's own";
  return "";
}
function s3KeysOf(f) {
  const k = {
    accessKeyId: f.accessKeyId.trim(),
    secretAccessKey: f.secretAccessKey.trim(),
    region: f.region.trim(),
    endpoint: f.endpoint.trim()
  };
  return k.accessKeyId || k.secretAccessKey ? k : void 0;
}
function readOnly(r) {
  if (!r.user) return "read-only: the demo was started without --user";
  if (!r.admin) return "read-only: the server was started without --admin or --zoo";
  return "";
}
function redirected(search2) {
  const q = new URLSearchParams(search2);
  const state = q.get("state") ?? "";
  if (!state) return null;
  return { code: q.get("code") ?? "", state, error: q.get("error") ?? "" };
}

// demo/web/src/tabs/sources.ts
var note2 = () => $("sources-note");
var saidWhy = "";
function sourcesTable(r, changed2) {
  const why = readOnly(r);
  if (!r.user) return el("p", { class: "muted", text: why });
  if (!r.sources.length) return el("p", { class: "muted", text: `${r.user} has no sources: installs read the catalog's own.` });
  return table(
    ["name", "address", "signs in with", ""],
    r.sources.map((s2) => {
      let control = "";
      if (!why) {
        control = confirmed("Remove", `Remove ${s2.name}?`, async () => {
          const gone = await catalog.removeModelSource({ name: s2.name });
          await changed2();
          say(note2(), gone.wasRegistered ? `Removed ${s2.name}.` : `${s2.name} was not registered.`);
        }, off(TurboCatalogService.method.removeModelSource));
      }
      return [s2.name, el("code", { text: s2.address }), signInText(s2.signIn), control];
    })
  );
}
async function act(change, after2) {
  try {
    say(note2(), "working", "busy");
    const done = await change();
    await after2();
    say(note2(), done);
  } catch (e) {
    say(note2(), explain(e));
  }
}
var freshLoad = latest();
async function load() {
  const mine = freshLoad();
  try {
    const r = await catalog.listModelSources({});
    if (!mine()) return;
    $("sources-list").replaceChildren(sourcesTable(r, load));
    const why = readOnly(r);
    $("sources-form").hidden = why !== "";
    $("sources-drive").hidden = why !== "" || !r.driveSignIn;
    if (why) say(note2(), why);
    else if (saidWhy && note2().textContent === saidWhy) say(note2(), "");
    saidWhy = why;
  } catch (e) {
    if (!mine()) return;
    $("sources-list").replaceChildren(reachable() === false ? el("p", { class: "error", text: "The server is not reachable, so the sources cannot be listed." }) : failed(explain(e)));
    say(note2(), "");
  }
}
function add(ev) {
  ev.preventDefault();
  const name = input("sources-name").value.trim();
  const address2 = input("sources-address").value.trim();
  const apiKey = input("sources-key").value.trim();
  const s3Keys = s3KeysOf({
    accessKeyId: input("sources-s3-key-id").value,
    secretAccessKey: input("sources-s3-secret").value,
    region: input("sources-s3-region").value,
    endpoint: input("sources-s3-endpoint").value
  });
  const register = async () => {
    const r = await catalog.registerModelSource({ name, address: address2, apiKey, s3Keys });
    input("sources-key").value = "";
    input("sources-s3-key-id").value = "";
    input("sources-s3-secret").value = "";
    return `Registered ${r.source?.name ?? name}.`;
  };
  void busy(submitter(ev), () => act(register, load));
}
function connect() {
  const name = input("sources-drive-name").value.trim() || "Google Drive";
  const folder = input("sources-drive-folder").value.trim() || "root";
  const address2 = folder.startsWith("gdrive://") ? folder : `gdrive://${folder}`;
  return act(async () => {
    const r = await catalog.startDriveSignIn({ name, address: address2 });
    location.assign(r.authorizeUrl);
    return "Opening Google sign-in.";
  }, async () => void 0);
}
async function finish(back) {
  history.replaceState(null, "", `${location.pathname}#models`);
  if (back.error) {
    await load();
    say(note2(), `Google sign-in was not completed: ${back.error}`, "error");
    return;
  }
  await act(async () => {
    const r = await catalog.finishDriveSignIn({ state: back.state, code: back.code });
    return `Signed in: ${r.source?.name ?? "the Drive"} is registered.`;
  }, load);
}
function init8() {
  $("sources-form").addEventListener("submit", add);
  lock($("sources-form"), TurboCatalogService.method.registerModelSource);
  lock($("sources-drive"), TurboCatalogService.method.startDriveSignIn);
  const drive = $("sources-connect");
  drive.addEventListener("click", () => void busy(drive, connect));
  const again = $("sources-refresh");
  again.addEventListener("click", () => void busy(again, load));
  const back = redirected(location.search);
  let pending = back;
  let finishing = false;
  onServed(() => {
    if (pending) {
      const b = pending;
      pending = null;
      finishing = true;
      void finish(b).finally(() => finishing = false);
    } else if (!finishing) void load();
  });
  onChange(() => {
    if (!pending && !finishing) void load();
  });
  return back !== null;
}

// demo/web/src/tabs/models.ts
var installing = null;
var saving = null;
var trainingRun = null;
var filterKind = "";
var filterText = "";
var note3 = () => $("models-note");
function bar(done, total) {
  const p = el("progress", { max: String(Math.max(1, total)) });
  p.value = Math.min(done, Math.max(1, total));
  return p;
}
async function act2(change) {
  try {
    const done = await change();
    await loadCatalog();
    say(note3(), done);
  } catch (e) {
    say(note3(), explain(e));
  }
}
function entryRow(e, w) {
  const terms = e.missingTerms.map((t) => termsBlock(e, t, w, (t2) => act2(() => accept(e, t2))));
  let install = "";
  if (w.canChange) {
    const b = el("button", { type: "button", class: e.missingTerms.length ? "" : "primary", text: e.installed ? "Reinstall" : "Install" });
    const why = off(TurboCatalogService.method.installModel);
    b.disabled = e.missingTerms.length > 0 || why !== "";
    b.title = e.missingTerms.length > 0 ? "Accept the licence terms first" : why;
    b.addEventListener("click", () => void busy(b, () => installModel(e)));
    install = b;
  }
  const licence = el("span", { class: "licence", text: e.licence }, nonCommercial(e) ? el("strong", { class: "error", text: " Non-commercial use only" }) : null);
  return el(
    "div",
    { class: "entry" },
    el("div", { class: "id", text: e.id }),
    el("div", { class: "purpose", text: purpose(e) }),
    el(
      "div",
      { class: "facts" },
      el("span", { class: "tag kind", text: kindName(e.kind) }),
      e.language ? el("span", { class: "tag", title: "language", text: e.language }) : null,
      e.installed ? el("span", { class: "tag installed", text: "installed" }) : null
    ),
    el("div", { class: "facts" }, licence, el("span", { class: "muted", text: size(e.totalBytes) })),
    el("div", { class: "acts" }, install, saveButtons(e, w), terms)
  );
}
function catalogTable(entries, w) {
  if (!entries.length) return el("p", { class: "muted", text: "The catalog lists nothing." });
  const rows2 = entries.map((e) => entryRow(e, w));
  const list5 = el("div", { class: "catalog" }, rows2);
  const count2 = el("span", { class: "muted", id: "models-count", role: "status" });
  const box5 = el("input", { id: "models-filter", type: "search", placeholder: "filter by name, purpose, kind, language or licence", autocomplete: "off", "aria-label": "Filter the catalog" });
  box5.value = filterText;
  if (filterKind && !entries.some((e) => e.kind === filterKind)) filterKind = "";
  const chips = [["", "all", entries.length], ...kinds(entries).map(([k, n2]) => [k, kindName(k), n2])].map(([k, name, n2]) => {
    const chip = el("button", { type: "button", class: "chip", "data-kind": k, text: `${name} ${n2}` });
    chip.addEventListener("click", () => {
      filterKind = k;
      apply2();
    });
    return [k, chip];
  });
  const none = el("p", { class: "muted", text: "No entry matches.", hidden: "" });
  function apply2() {
    let shown2 = 0;
    entries.forEach((e, i) => {
      const on = matches(e, filterKind, filterText);
      rows2[i].hidden = !on;
      if (on) shown2++;
    });
    for (const [k, chip] of chips) {
      chip.classList.toggle("on", k === filterKind);
      chip.setAttribute("aria-pressed", String(k === filterKind));
    }
    count2.textContent = shown2 === entries.length ? `${entries.length} entries` : `${shown2} of ${entries.length} entries`;
    list5.hidden = shown2 === 0;
    none.hidden = shown2 !== 0;
  }
  box5.addEventListener("input", () => {
    filterText = box5.value;
    apply2();
  });
  apply2();
  return el("div", {}, el("div", { class: "row filter" }, box5, count2), el("div", { class: "chips" }, chips.map(([, c]) => c)), list5, none);
}
function saveButtons(e, w) {
  if (!w.user || e.missingTerms.length > 0) return null;
  if (!e.files.length) return null;
  return el(
    "details",
    { class: "files" },
    el("summary", { text: e.files.length === 1 ? "1 file" : `${e.files.length} files` }),
    e.files.map((f) => {
      const b = el("button", { type: "button", text: `Save ${f.path}`, title: size(f.size) });
      b.addEventListener("click", () => void busy(b, () => saveFile(e, f)));
      return b;
    })
  );
}
async function saveFile(e, f) {
  saving?.abort();
  const ac = new AbortController();
  saving = ac;
  const status3 = el("p", { class: "muted", text: `Reading ${e.id}/${f.path}\u2026` });
  $("models-install").replaceChildren(el("h3", { text: `Save ${e.id}/${f.path}` }), status3);
  try {
    await download2(e, f, status3, ac.signal);
  } catch (err) {
    status3.className = "error";
    status3.textContent = describe(err);
  } finally {
    if (saving === ac) saving = null;
  }
}
var freshCatalog = latest();
async function loadCatalog() {
  const mine = freshCatalog();
  const box5 = $("models-catalog");
  say(note3(), "Reading the catalog\u2026", "busy");
  try {
    const [cat, installed, acc] = await Promise.all([catalog.listModelCatalog({}), catalog.listInstalledModels({}), catalog.listLicenceAcceptances({})]);
    if (!mine()) return;
    const w = who(acc);
    box5.replaceChildren(
      el("h3", { text: "Installed" }),
      table(["id", "version", "files", "size", "directory"], installed.models.map((m) => [m.id, m.version, String(m.files.length), size(m.totalBytes), m.dir]), "", "Nothing is installed."),
      el("h3", { text: "Accepted terms" }),
      acceptedTable(acc, w, async (t) => {
        const done = await revoke(t);
        await loadCatalog();
        say(note3(), done);
      }),
      el("h3", { text: "Catalog" }),
      el("p", { class: "muted", text: w.user ? `Licence terms still to accept are those for ${w.user}; read them before accepting.${w.why ? ` (${w.why})` : ""}` : w.why }),
      catalogTable(cat.entries, w)
    );
    say(note3(), "");
  } catch (e) {
    if (!mine()) return;
    if (reachable() === false) say(note3(), "The server is not reachable, so its catalog cannot be read.", "error");
    else say(note3(), explain(e));
  }
}
async function installModel(e) {
  installing?.abort();
  const ac = new AbortController();
  installing = ac;
  const log = $("models-install");
  const status3 = el("p", { class: "muted", text: `Installing ${e.id}\u2026` });
  const progress = bar(0, 1);
  log.replaceChildren(el("h3", { text: `Install ${e.id}` }), progress, status3);
  try {
    for await (const u of catalog.installModel({ artifactId: e.id }, { signal: ac.signal })) {
      if (u.update.case === "progress") {
        const p = u.update.value;
        progress.max = Math.max(1, p.fileCount);
        progress.value = p.stage === 4 /* VERIFIED */ ? p.fileIndex + 1 : p.fileIndex;
        status3.textContent = `${InstallStage[p.stage]}: file ${p.fileIndex + 1} of ${p.fileCount} ${p.path} (${size(p.bytes)})`;
      } else if (u.update.case === "installed") {
        const m = u.update.value;
        progress.value = progress.max;
        status3.textContent = `Installed ${m.id} ${m.version}: ${m.files.length} files, ${size(m.totalBytes)} in ${m.dir}.`;
      }
    }
    await loadCatalog();
  } catch (err) {
    status3.className = "error";
    status3.textContent = describe(err);
  } finally {
    if (installing === ac) installing = null;
  }
}
var freshTraining = latest();
async function loadTraining() {
  const mine = freshTraining();
  const box5 = $("models-static");
  try {
    const t = await training.listTeachers({});
    if (!mine()) return;
    options($("train-teacher"), t.teachers.map((x) => x.name), t.teachers.map((x) => `${x.name} (hidden ${x.hidden}, ${x.vocabSize} words)`), "no teacher is served");
    let made;
    try {
      const s2 = await training.listStaticModels({});
      if (!mine()) return;
      made = table(
        ["name", "dims", "words", "learned", "dtype", "teacher", "quality", "served", ""],
        s2.models.map((m) => [
          m.name,
          String(m.dims),
          String(m.vocabSize),
          String(m.learnedTerms),
          m.dtype,
          m.teacher?.modelId ?? "",
          m.quality ? `top-1 ${m.quality.staticTop1.toFixed(3)} of ${m.quality.baseTop1.toFixed(3)}, rho ${m.quality.similaritySpearman.toFixed(3)}` : "",
          m.served ? "yes" : m.problem,
          deleteButton(m.name)
        ])
      );
    } catch (e) {
      if (!mine()) return;
      made = reachable() === false ? el("p", { class: "error", text: "The server is not reachable, so its static models cannot be listed." }) : failed(explain(e));
    }
    box5.replaceChildren(el("h3", { text: "Teachers" }), table(["name", "model", "hidden", "words"], t.teachers.map((x) => [x.name, `${x.modelId}@${x.revision}`, String(x.hidden), String(x.vocabSize)])), el("h3", { text: "Static models" }), made);
  } catch (e) {
    if (!mine()) return;
    const down = reachable() === false;
    options($("train-teacher"), [], void 0, down ? "the server is not reachable" : "the teachers could not be listed");
    box5.replaceChildren(down ? el("p", { class: "error", text: "The server is not reachable, so its teachers and static models cannot be listed." }) : failed(explain(e)));
  }
}
function deleteButton(name) {
  return confirmed("Delete", `Delete ${name}?`, async () => {
    await training.deleteStaticModel({ name });
    await Promise.all([loadTraining(), refresh()]);
  }, off(TurboTrainingService.method.deleteStaticModel));
}
async function train(ev) {
  ev.preventDefault();
  await busy(submitter(ev), async () => {
    stopTraining();
    const ac = new AbortController();
    trainingRun = ac;
    const steps = $("train-steps");
    const progress = bar(0, 1);
    steps.replaceChildren();
    $("train-progress").replaceChildren(progress);
    say($("train-note"), "Training\u2026", "busy");
    const t0 = performance.now();
    let last2 = -1;
    try {
      const req = {
        name: input("train-name").value.trim(),
        teacher: $("train-teacher").value,
        pcaDims: parseInt(input("train-dims").value, 10) || 0,
        dtype: $("train-dtype").value === "F16" ? 2 /* F16 */ : 1 /* F32 */,
        vocabularyArtifactId: input("train-vocabulary").value.trim(),
        terms: $("train-terms").value.split("\n").map((t) => t.trim()).filter((t) => t.length)
      };
      for await (const u of training.trainStaticModel(req, { signal: ac.signal })) {
        const ms = `${((performance.now() - t0) / 1e3).toFixed(1)} s`;
        if (u.update.case === "progress") {
          const p = u.update.value;
          progress.max = Math.max(1, Number(p.totalItems));
          progress.value = Number(p.completedItems);
          const facts2 = [
            p.totalItems ? `${p.completedItems} of ${p.totalItems}` : "",
            p.addedTerms || p.refusedTerms ? `${p.addedTerms} terms added, ${p.refusedTerms} refused` : "",
            p.explainedVariance ? `explained variance ${(p.explainedVariance * 100).toFixed(1)}%` : "",
            p.quality ? `top-1 ${p.quality.staticTop1.toFixed(3)} (teacher ${p.quality.baseTop1.toFixed(3)}), rho ${p.quality.similaritySpearman.toFixed(3)}` : ""
          ].filter(Boolean);
          const line = el("li", {}, el("span", { class: "badge", text: ms }), ` ${TrainStage[p.stage]}${facts2.length ? `: ${facts2.join(", ")}` : ""}`);
          if (p.stage === last2 && steps.lastElementChild) steps.lastElementChild.replaceWith(line);
          else steps.append(line);
          last2 = p.stage;
        } else if (u.update.case === "trained") {
          const t = u.update.value;
          progress.value = progress.max;
          const refused = t.refusedTerms.map((r) => `${r.term} (${r.reason})`).join(", ");
          steps.append(el("li", {}, el("span", { class: "badge", text: ms }), ` trained ${t.model?.name ?? ""}: ${t.model?.dims ?? 0} dims, ${t.model?.vocabSize ?? 0} words${t.addedTerms.length ? `, added ${t.addedTerms.join(" ")}` : ""}${refused ? `; refused ${refused}` : ""}`));
        }
      }
      await Promise.all([loadTraining(), refresh()]);
      say($("train-note"), "");
    } catch (e) {
      say($("train-note"), explain(e));
    } finally {
      if (trainingRun === ac) trainingRun = null;
    }
  }, $("train-stop"));
}
function stopTraining() {
  trainingRun?.abort();
  trainingRun = null;
}
function stop3() {
  installing?.abort();
  installing = null;
  saving?.abort();
  saving = null;
  stopTraining();
}
function init9() {
  const read = $("models-catalog-load");
  read.addEventListener("click", () => void busy(read, loadCatalog));
  $("train-form").addEventListener("submit", train);
  lock($("train-form"), TurboTrainingService.method.trainStaticModel);
  $("train-stop").addEventListener("click", stopTraining);
  const again = $("train-refresh");
  again.addEventListener("click", () => void busy(again, loadTraining));
  let reading = false;
  let more = false;
  const teachers = () => {
    if (reading) return void (more = true);
    reading = true;
    void loadTraining().finally(() => {
      reading = false;
      if (more) {
        more = false;
        teachers();
      }
    });
  };
  onServed(teachers);
  onCapabilities(teachers);
  onChange(() => {
    teachers();
    if ($("models-catalog").childElementCount) void loadCatalog();
  });
  init8();
}

// demo/web/src/viz/link.ts
function linker() {
  const focus = { chunk: null, term: null };
  const listeners3 = [];
  const tell = () => listeners3.forEach((l) => l(focus));
  return {
    on(l) {
      listeners3.push(l);
      l(focus);
    },
    chunk(i) {
      if (focus.chunk === i) return;
      focus.chunk = i;
      tell();
    },
    term(t) {
      if (focus.term === t) return;
      focus.term = t;
      tell();
    }
  };
}

// demo/web/src/viz/model.ts
function leaves(units) {
  if (!units.length) return [];
  const finest = Math.min(...units.map((u) => u.level));
  return units.filter((u) => u.level === finest).sort((a, b) => a.start - b.start || a.end - b.end);
}
function regions(units, leafUnits) {
  if (!leafUnits.length) return [];
  const finest = leafUnits[0].level;
  const out = [];
  for (const u of units) {
    if (u.level === finest) continue;
    const members = leafUnits.flatMap((l, i) => l.start >= u.start && l.end <= u.end && !l.empty ? [i] : []);
    if (members.length >= 2 && members.length < leafUnits.filter((l) => !l.empty).length) out.push({ unit: u, members });
  }
  return out.sort((a, b) => a.unit.start - b.unit.start);
}
function hull(pts) {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]).filter((q, i, all) => i === 0 || q[0] !== all[i - 1][0] || q[1] !== all[i - 1][1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half2 = (src) => {
    const h = [];
    for (const q of src) {
      while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], q) <= 0) h.pop();
      h.push(q);
    }
    h.pop();
    return h;
  };
  return [...half2(p), ...half2([...p].reverse())];
}
function scale(lo, hi, a, b) {
  return hi === lo ? () => (a + b) / 2 : (v) => a + (v - lo) / (hi - lo) * (b - a);
}
function spread(chunks) {
  const seen = /* @__PURE__ */ new Map();
  for (const c of chunks) for (const t of new Set(c.map((x) => x.term))) seen.set(t, (seen.get(t) ?? 0) + 1);
  return seen;
}
function telling(here, there, listedIn, count2) {
  const rare = count2 > 1 ? Math.log(count2 / Math.max(1, listedIn)) : 1;
  return here * Math.log(here / there) * rare;
}
function distinctive(chunk, chunkTotal, doc2, docTotal, k, listed2 = /* @__PURE__ */ new Map(), count2 = 1) {
  if (!chunk.length || chunkTotal <= 0 || docTotal <= 0) return [];
  const shares = new Map(doc2.map((t) => [t.term, t.weight / docTotal]));
  const floor = doc2.length ? Math.min(...doc2.map((t) => t.weight / docTotal)) : 1;
  return chunk.map((t) => ({ term: t.term, score: telling(t.weight / chunkTotal, shares.get(t.term) ?? floor, listed2.get(t.term) ?? 1, count2) })).filter((t) => t.score > 0).sort((a, b) => b.score - a.score || (a.term < b.term ? -1 : 1)).slice(0, k).map((t) => t.term);
}
function riverTerms(doc2, docTotal, chunks, k, uneven) {
  let picked2;
  if (uneven) {
    const shares = new Map(doc2.map((t) => [t.term, t.weight / docTotal]));
    const listed2 = spread(chunks.map((c) => c.terms));
    const best2 = /* @__PURE__ */ new Map();
    for (const c of chunks) {
      for (const t of c.terms) {
        const there = shares.get(t.term);
        if (there === void 0 || c.total <= 0) continue;
        best2.set(t.term, Math.max(best2.get(t.term) ?? 0, telling(t.weight / c.total, there, listed2.get(t.term) ?? 1, chunks.length)));
      }
    }
    picked2 = [...best2].filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, k).map(([t]) => t);
  } else {
    picked2 = [...doc2].sort((a, b) => b.weight - a.weight).slice(0, k).map((t) => t.term);
  }
  const peak = (term) => {
    const w = chunks.map((c) => c.terms.find((t) => t.term === term)?.weight ?? 0);
    return w.indexOf(Math.max(...w));
  };
  return picked2.map((t, i) => ({ t, i, at: peak(t) })).sort((a, b) => a.at - b.at || a.i - b.i).map((x) => x.t);
}
function streams(weights) {
  const cols = weights[0]?.length ?? 0;
  const out = weights.map(() => ({ lo: new Array(cols).fill(0), hi: new Array(cols).fill(0) }));
  for (let c = 0; c < cols; c++) {
    let at = -weights.reduce((s2, w) => s2 + w[c], 0) / 2;
    weights.forEach((w, t) => {
      out[t].lo[c] = at;
      at += w[c];
      out[t].hi[c] = at;
    });
  }
  return out;
}
function pieces(text, chunks, marks) {
  const clamp = (v) => Math.max(0, Math.min(text.length, v));
  const edges = /* @__PURE__ */ new Set([0, text.length]);
  for (const c of chunks) edges.add(clamp(c.start)).add(clamp(c.end));
  for (const [s2, e] of marks) edges.add(clamp(s2)).add(clamp(e));
  const cuts = [...edges].sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i + 1 < cuts.length; i++) {
    const [s2, e] = [cuts[i], cuts[i + 1]];
    if (e <= s2) continue;
    const chunk = chunks.findIndex((c) => s2 >= c.start && e <= c.end);
    out.push({ text: text.slice(s2, e), start: s2, chunk: chunk < 0 ? null : chunk, marked: marks.some(([a, b]) => s2 >= a && e <= b) });
  }
  return out;
}
function aroundGap(text, gaps, i) {
  const at = gaps[i];
  const from2 = i > 0 ? gaps[i - 1] : 0;
  const to2 = i + 1 < gaps.length ? gaps[i + 1] : text.length;
  return [text.slice(from2, at).trim(), text.slice(at, to2).trim()];
}
function cutReason(code) {
  return ["document start", "sentence end", "section", "heading", "paragraph", "topic shift", "clause", "list item", "inside a sentence", "overlap"][code] ?? `reason ${code}`;
}

// demo/web/src/viz/svg.ts
var ns = null;
function namespace() {
  if (ns === null) {
    const t = document.createElement("template");
    t.innerHTML = "<svg></svg>";
    ns = t.content.firstElementChild.namespaceURI;
  }
  return ns;
}
function s(tag, attrs = {}, ...children) {
  const e = document.createElementNS(namespace(), tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== void 0) e.setAttribute(k, String(v));
  for (const c of children) if (c !== null) e.append(c);
  return e;
}
function figure(w, h, label) {
  return s("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": label, class: "viz" });
}
function n(v) {
  return String(Math.round(v * 100) / 100);
}
function ordered(i, count2) {
  const share = count2 > 1 ? Math.round(100 * i / (count2 - 1)) : 50;
  return `color-mix(in oklab, var(--seq-from), var(--seq-to) ${share}%)`;
}

// demo/web/src/tabs/chunktext.ts
function chunkText(text, chunks, link, marks) {
  const box5 = el("div", { class: "doc chunked", tabindex: "0", "aria-label": "The document, by chunk" });
  let drawn;
  link.on((f) => {
    if (f.term !== drawn) {
      drawn = f.term;
      const seen = /* @__PURE__ */ new Set();
      box5.replaceChildren(
        ...pieces(text, chunks, f.term === null ? [] : marks(f.term)).map((p) => {
          const body2 = p.marked ? el("mark", { class: "occ", text: p.text }) : p.text;
          if (p.chunk === null) return el("span", {}, body2);
          const first = !seen.has(p.chunk);
          seen.add(p.chunk);
          const span = el("span", { class: "ck", "data-chunk": String(p.chunk), style: `--ck: ${ordered(p.chunk, chunks.length)}` }, first ? el("sup", { text: String(p.chunk + 1) }) : null, body2);
          span.addEventListener("mouseenter", () => link.chunk(p.chunk));
          return span;
        })
      );
    }
    for (const span of box5.querySelectorAll(".ck")) span.classList.toggle("lit", span.dataset.chunk === String(f.chunk));
  });
  box5.addEventListener("mouseleave", () => link.chunk(null));
  return box5;
}
function follow(box5, link) {
  link.on((f) => {
    if (f.chunk === null || box5.matches(":hover")) return;
    const span = box5.querySelector(`.ck[data-chunk="${f.chunk}"]`);
    if (span) box5.scrollTop = Math.max(0, span.offsetTop - box5.offsetTop - 12);
  });
}

// demo/web/src/tabs/meaningmap.ts
var W = 640;
var H = 340;
var PAD = { l: 34, r: 16, t: 16, b: 34 };
function meaningMap(m, link) {
  const placed = m.chunks.map((c, i) => ({ c, i })).filter((p) => !p.c.empty);
  const xs = placed.map((p) => p.c.x);
  const ys = placed.map((p) => p.c.y);
  const zs = placed.map((p) => p.c.z);
  const sx = scale(Math.min(...xs), Math.max(...xs), PAD.l + 24, W - PAD.r - 24);
  const sy = scale(Math.min(...ys), Math.max(...ys), H - PAD.b - 24, PAD.t + 24);
  const sz = scale(Math.min(...zs), Math.max(...zs), 5, 11);
  const at = new Map(placed.map((p) => [p.i, [sx(p.c.x), sy(p.c.y)]]));
  const pct = (v) => v === void 0 ? "" : ` (${(v * 100).toFixed(0)}% of the variance)`;
  const svg = figure(W, H, `Meaning map of ${placed.length} chunks`);
  svg.append(
    s("rect", { x: PAD.l, y: PAD.t, width: W - PAD.l - PAD.r, height: H - PAD.t - PAD.b, class: "frame" }),
    s("text", { x: (PAD.l + W - PAD.r) / 2, y: H - 10, class: "axis", "text-anchor": "middle" }, `first direction of meaning${pct(m.variance[0])}`),
    s("text", { x: 12, y: (PAD.t + H - PAD.b) / 2, class: "axis", "text-anchor": "middle", transform: `rotate(-90 12 ${(PAD.t + H - PAD.b) / 2})` }, `second direction${pct(m.variance[1])}`)
  );
  for (const r of regions(m.units, m.chunks)) {
    const pts = r.members.flatMap((i) => at.has(i) ? [at.get(i)] : []);
    const h = hull(pts);
    const g = s("g", { class: "region" });
    if (h.length >= 3) g.append(s("polygon", { points: h.map(([x, y]) => `${n(x)},${n(y)}`).join(" ") }));
    else if (h.length === 2) g.append(s("line", { x1: n(h[0][0]), y1: n(h[0][1]), x2: n(h[1][0]), y2: n(h[1][1]) }));
    g.append(s("title", {}, `Level ${r.unit.level} unit ${r.unit.item}: chunks ${r.members[0] + 1} to ${r.members[r.members.length - 1] + 1}`));
    svg.append(g);
  }
  if (placed.length > 1) svg.append(s("polyline", { class: "trail", points: placed.map((p) => at.get(p.i).map(n).join(",")).join(" ") }));
  const info = el("p", { class: "info", role: "status", text: "Point at a dot to read its chunk; the same chunk lights up in the text." });
  const dots = /* @__PURE__ */ new Map();
  const few = placed.length <= 14;
  for (const p of placed) {
    const [x, y] = at.get(p.i);
    const r = m.third ? sz(p.c.z) : 7;
    const names2 = m.terms(p.i);
    const left = x > W * 0.68;
    const label = `${p.i + 1}${few && names2.length ? ` \xB7 ${names2.slice(0, 2).join(", ")}` : ""}`;
    const g = s(
      "g",
      { class: "dot", tabindex: 0, role: "button", "aria-label": `Chunk ${p.i + 1} of ${m.chunks.length}${names2.length ? `: ${names2.join(", ")}` : ""}` },
      // A target larger than the mark.
      s("circle", { cx: n(x), cy: n(y), r: 18, class: "hit" }),
      s("circle", { cx: n(x), cy: n(y), r: n(r), class: "mark", style: `fill: ${ordered(p.i, m.chunks.length)}` }),
      s("text", { x: n(x + (left ? -r - 5 : r + 5)), y: n(y + 4), class: "label", "text-anchor": left ? "end" : "start" }, label)
    );
    const point = () => link.chunk(p.i);
    g.addEventListener("mouseenter", point);
    g.addEventListener("focus", point);
    g.addEventListener("mouseleave", () => link.chunk(null));
    g.addEventListener("blur", () => link.chunk(null));
    dots.set(p.i, g);
    svg.append(g);
  }
  link.on((f) => {
    for (const [i, g] of dots) g.classList.toggle("lit", i === f.chunk);
    svg.classList.toggle("dim", f.chunk !== null);
    if (f.chunk === null) return;
    const c = m.chunks[f.chunk];
    const cos = m.toNext(f.chunk);
    const names2 = m.terms(f.chunk);
    info.textContent = `Chunk ${f.chunk + 1} of ${m.chunks.length}, characters ${c.start} to ${c.end}${names2.length ? `, told apart by ${names2.join(", ")}` : ""}${cos === void 0 ? "" : `, cosine to the next ${cos.toFixed(3)}`}: ${short(m.text.slice(c.start, c.end), 140)}`;
  });
  const others = m.units.length - m.chunks.length;
  const legend2 = el(
    "div",
    { class: "legend" },
    el("span", { class: "swatch ramp-key" }, el("i", {}), " a chunk, coloured from the first (light) to the last (dark)"),
    el("span", { class: "swatch trail-key" }, el("i", {}), " reading order"),
    m.third ? el("span", { class: "muted", text: "size: the third direction" }) : null,
    others > 0 ? el("span", { class: "muted", text: `${others} coarser ${others === 1 ? "unit is" : "units are"} not drawn as dots; one holding some of the chunks is a shaded region` }) : null
  );
  return el("div", { class: "figure" }, el("div", { class: "scroll" }, svg), legend2, info);
}

// demo/web/src/tabs/river.ts
var W2 = 640;
var H2 = 210;
var PAD2 = { l: 12, r: 12, t: 22, b: 10 };
function river(m, link) {
  const x = scale(0, Math.max(1, m.length), PAD2.l, W2 - PAD2.r);
  const edges = streams(m.weights);
  const reach = Math.max(1e-9, ...edges.flatMap((e) => e.hi), ...edges.flatMap((e) => e.lo.map((v) => -v)));
  const y = scale(-reach, reach, H2 - PAD2.b, PAD2.t);
  const svg = figure(W2, H2, `Term river: ${m.terms.join(", ")}`);
  const stops2 = m.chunks.flatMap((c) => [x(c.start + (c.end - c.start) * 0.25), x(c.start + (c.end - c.start) * 0.75)]);
  const along = (v) => {
    const pts = v.flatMap((w, c) => [[stops2[2 * c], y(w)], [stops2[2 * c + 1], y(w)]]);
    return pts.length ? [[x(m.chunks[0].start), pts[0][1]], ...pts, [x(m.chunks[m.chunks.length - 1].end), pts[pts.length - 1][1]]] : pts;
  };
  const curve = (pts, first) => pts.map(([px, py], i) => {
    if (i === 0) return `${first}${n(px)} ${n(py)}`;
    const [qx, qy] = pts[i - 1];
    const mid = (qx + px) / 2;
    return `C${n(mid)} ${n(qy)} ${n(mid)} ${n(py)} ${n(px)} ${n(py)}`;
  }).join(" ");
  m.chunks.forEach((c, i) => {
    if (i) svg.append(s("line", { x1: n(x(c.start)), x2: n(x(c.start)), y1: PAD2.t - 4, y2: H2 - PAD2.b, class: "cut-line" }));
    svg.append(s("text", { x: n(x((c.start + c.end) / 2)), y: 12, class: "tick", "text-anchor": "middle" }, String(i + 1)));
  });
  const paths = m.terms.map((term, t) => {
    const up = along(edges[t].hi);
    const down = along(edges[t].lo).reverse();
    const p = s("path", { d: `${curve(up, "M")} ${curve(down, "L")} Z`, class: "stream", style: `fill: var(--m${t})`, tabindex: 0, role: "button", "aria-label": `${term}, ${m.occurrences(term)} occurrences` }, s("title", {}, term));
    const on = () => link.term(term);
    p.addEventListener("mouseenter", on);
    p.addEventListener("focus", on);
    svg.append(p);
    const widest = m.weights[t].indexOf(Math.max(...m.weights[t]));
    const thick = y(edges[t].lo[widest]) - y(edges[t].hi[widest]);
    if (thick >= 14 && m.chunks[widest]) {
      const c = m.chunks[widest];
      svg.append(s("text", { x: n(x((c.start + c.end) / 2)), y: n((y(edges[t].lo[widest]) + y(edges[t].hi[widest])) / 2 + 4), class: "stream-name", "text-anchor": "middle" }, term));
    }
    return p;
  });
  const IDLE = "Point at a stream or a name to mark that term in the text.";
  const info = el("p", { class: "info", role: "status", text: IDLE });
  const keys = m.terms.map((term, t) => {
    const b = el("button", { type: "button", class: "swatch key", style: `--mark: var(--m${t})` }, el("i", {}), ` ${term} `, el("span", { class: "muted", text: String(m.occurrences(term)) }));
    b.addEventListener("mouseenter", () => link.term(term));
    b.addEventListener("focus", () => link.term(term));
    b.addEventListener("click", () => link.term(term));
    return b;
  });
  const clear = el("button", { type: "button", class: "small", text: "Clear" });
  clear.addEventListener("click", () => link.term(null));
  link.on((f) => {
    const at = f.term === null ? -1 : m.terms.indexOf(f.term);
    paths.forEach((p, t) => p.classList.toggle("faded", at >= 0 && t !== at));
    keys.forEach((k, t) => k.classList.toggle("on", t === at));
    clear.hidden = at < 0;
    if (at >= 0) {
      const where = m.weights[at].flatMap((w, c) => w > 0 ? [c + 1] : []);
      info.textContent = `"${m.terms[at]}" occurs ${m.occurrences(m.terms[at])} times; it is among the listed terms of ${where.length === 1 ? "chunk" : "chunks"} ${where.join(", ")}. Its occurrences are marked in the text.`;
    } else info.textContent = IDLE;
  });
  return el("div", { class: "figure" }, el("div", { class: "scroll" }, svg), el("div", { class: "legend" }, keys, clear), info);
}

// demo/web/src/tabs/seismograph.ts
var W3 = 640;
var GUTTER = 150;
var RIGHT = 12;
var BAND = 14;
var CURVE = 120;
var ROW = 20;
function seismograph(m, link) {
  const len = Math.max(1, m.text.length);
  const x = scale(0, len, GUTTER, W3 - RIGHT);
  const top = 8;
  const curveTop = top + BAND + 10;
  const base2 = curveTop + CURVE;
  const rowsTop = base2 + 26;
  const H3 = rowsTop + (m.strategies.length + 1) * ROW + 6;
  const hasCosine = m.gaps.some((g) => g.smoothed !== void 0);
  const svg = figure(W3, H3, `Topic seismograph: ${m.gaps.length} sentence gaps, ${m.gaps.filter((g) => g.cut).length} cuts`);
  svg.append(s("text", { x: GUTTER - 8, y: top + BAND - 3, class: "row-name", "text-anchor": "end" }, "chunks"));
  const bands = m.chunks.map((c, i) => {
    const r = s("rect", { x: n(x(c.start) + 1), y: top, width: n(Math.max(1, x(c.end) - x(c.start) - 2)), height: BAND, rx: 3, class: "band", style: `fill: ${ordered(i, m.chunks.length)}` }, s("title", {}, `Chunk ${i + 1}`));
    r.addEventListener("mouseenter", () => link.chunk(i));
    r.addEventListener("mouseleave", () => link.chunk(null));
    return r;
  });
  svg.append(...bands);
  link.on((f) => bands.forEach((b, i) => b.classList.toggle("lit", i === f.chunk)));
  const values = m.gaps.map((g) => hasCosine ? g.smoothed : g.cost);
  const known = values.filter((v) => v !== void 0);
  const lo = hasCosine ? Math.min(...known) : 0;
  const hi = Math.max(...known, lo + 1e-9);
  const y = scale(lo, hi, base2, curveTop + 6);
  svg.append(
    s("line", { x1: GUTTER, x2: W3 - RIGHT, y1: base2, y2: base2, class: "axis-line" }),
    s("text", { x: GUTTER - 8, y: curveTop + 10, class: "tick", "text-anchor": "end" }, hasCosine ? hi.toFixed(2) : String(Math.round(hi))),
    s("text", { x: GUTTER - 8, y: base2, class: "tick", "text-anchor": "end" }, hasCosine ? lo.toFixed(2) : "0"),
    s("text", { x: GUTTER - 8, y: (curveTop + base2) / 2 + 4, class: "row-name", "text-anchor": "end" }, hasCosine ? "cosine across the gap" : "cost of a cut here")
  );
  if (hasCosine) {
    const pts = m.gaps.flatMap((g) => g.smoothed === void 0 ? [] : [[x(g.offset), y(g.smoothed)]]);
    if (pts.length > 1) {
      const line = pts.map(([px, py], i) => `${i ? "L" : "M"}${n(px)} ${n(py)}`).join(" ");
      svg.append(s("path", { d: `${line} L${n(pts[pts.length - 1][0])} ${base2} L${n(pts[0][0])} ${base2} Z`, class: "area" }), s("path", { d: line, class: "line" }));
    }
    const raw = m.gaps.flatMap((g) => g.raw === void 0 ? [] : [[x(g.offset), y(Math.max(lo, Math.min(hi, g.raw)))]]);
    if (raw.length > 1) svg.append(s("path", { d: raw.map(([px, py], i) => `${i ? "L" : "M"}${n(px)} ${n(py)}`).join(" "), class: "line raw" }));
  } else {
    for (const g of m.gaps) svg.append(s("line", { x1: n(x(g.offset)), x2: n(x(g.offset)), y1: base2, y2: n(y(g.cost)), class: g.cut ? "stem cut" : "stem" }));
  }
  for (const g of m.gaps) {
    if (!g.cut) continue;
    const gx = x(g.offset);
    const gy = y(hasCosine ? g.smoothed ?? lo : g.cost);
    svg.append(s("line", { x1: n(gx), x2: n(gx), y1: curveTop, y2: base2, class: "cut-line" }), s("path", { d: `M${n(gx)} ${n(gy - 6)} L${n(gx + 5)} ${n(gy)} L${n(gx)} ${n(gy + 6)} L${n(gx - 5)} ${n(gy)} Z`, class: "cut-mark" }));
  }
  const profile = m.gaps.filter((g) => g.cut).map((g) => g.offset);
  const rows2 = [
    { name: "this profile", note: `${m.chunks.length} chunks`, cuts: m.chunks.slice(1).map((c) => c.start) },
    ...m.strategies.map((t) => ({ name: t.name, note: `${t.chunks} chunks, ${t.shared} cuts shared, ${t.apart} not`, cuts: t.cuts.filter((c) => c > 0) }))
  ];
  rows2.forEach((r, i) => {
    const ry = rowsTop + i * ROW;
    const near = (c) => profile.some((p) => Math.abs(p - c) <= 2) || m.chunks.some((k) => k.start === c);
    svg.append(
      s("text", { x: GUTTER - 8, y: ry + 12, class: "row-name", "text-anchor": "end" }, short(r.name, 20), s("title", {}, `${r.name}: ${r.note}`)),
      s("rect", { x: GUTTER, y: ry + 2, width: W3 - RIGHT - GUTTER, height: ROW - 6, rx: 3, class: "barcode" }),
      ...r.cuts.map((c) => s("rect", { x: n(x(c) - 1), y: ry + 2, width: 2, height: ROW - 6, class: i === 0 || near(c) ? "tick-shared" : "tick-apart" }))
    );
  });
  const cross = s("line", { y1: curveTop, y2: base2, class: "cross", visibility: "hidden" });
  const pad = s("rect", { x: GUTTER, y: curveTop, width: W3 - RIGHT - GUTTER, height: CURVE, class: "pad", tabindex: 0, role: "slider", "aria-valuemin": 1, "aria-valuemax": m.gaps.length, "aria-label": "Sentence gaps; point or use the arrow keys" });
  svg.append(cross, pad);
  const info = el("p", { class: "info", role: "status", text: "Point at the curve to see the two sentences on either side of a gap." });
  const offsets = m.gaps.map((g) => g.offset);
  let picked2 = -1;
  const pick = (i) => {
    if (i < 0 || i >= m.gaps.length) return;
    picked2 = i;
    const g = m.gaps[i];
    cross.setAttribute("x1", n(x(g.offset)));
    cross.setAttribute("x2", n(x(g.offset)));
    cross.setAttribute("visibility", "visible");
    const [before, after2] = aroundGap(m.text, offsets, i);
    const value = g.smoothed !== void 0 ? `cosine ${g.smoothed.toFixed(3)}${g.depth > 0 ? `, depth ${g.depth.toFixed(3)}` : ""}` : `cost ${g.cost}`;
    info.replaceChildren(
      el("b", { text: `Gap ${i + 1} of ${m.gaps.length}${g.cut ? " (cut)" : ""}` }),
      `, ${cutReason(g.reason)}, ${value}. `,
      el("span", { class: "quote", text: short(before, 110) }),
      el("span", { class: "muted", text: g.cut ? "  \u2702  " : "  \xB7  " }),
      el("span", { class: "quote", text: short(after2, 110) || "(the end)" })
    );
    pad.setAttribute("aria-valuenow", String(i + 1));
    pad.setAttribute("aria-valuetext", `Gap ${i + 1} of ${m.gaps.length}`);
    const at = m.chunks.findIndex((c) => g.offset > c.start && g.offset <= c.end);
    link.chunk(at < 0 ? null : at);
  };
  pad.addEventListener("mousemove", (ev) => {
    const box5 = svg.getBoundingClientRect();
    const at = (ev.clientX - box5.left) / box5.width * W3;
    let best2 = 0;
    m.gaps.forEach((g, i) => {
      if (Math.abs(x(g.offset) - at) < Math.abs(x(m.gaps[best2].offset) - at)) best2 = i;
    });
    pick(best2);
  });
  pad.addEventListener("keydown", (ev) => {
    const step = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    ev.preventDefault();
    pick(Math.max(0, Math.min(m.gaps.length - 1, picked2 + step)));
  });
  pad.addEventListener("mouseleave", () => link.chunk(null));
  const cuts = m.gaps.filter((g) => g.cut).length;
  const legend2 = el(
    "div",
    { class: "legend" },
    el("span", { class: "swatch cut-key" }, el("i", {}), ` a gap the chunker cut at (${cuts})`),
    m.strategies.length ? el("span", { class: "swatch shared-key" }, el("i", {}), " a cut this profile also makes") : null,
    m.strategies.length ? el("span", { class: "swatch apart-key" }, el("i", {}), " a cut only that budget makes") : null
  );
  const how = hasCosine ? `The line is the cosine between the sentences on each side of a gap, smoothed over ${m.smoothing} (window ${m.window}); the thin line is the raw value. A valley is a change of topic.` : "This graph holds no sentence vectors, so there is no cosine to draw: each stem is what the chunker charges to cut at that gap, and a short stem is an easy cut (a paragraph or a heading).";
  return el("div", { class: "figure" }, el("p", { class: "muted", text: how }), el("div", { class: "scroll" }, svg), legend2, info);
}

// demo/web/src/tabs/projection.ts
var box3 = () => $("projection-out");
function budgets(s2) {
  const out = [];
  for (const item of s2.split(",")) {
    const [name, max, target] = item.split(":").map((x) => x.trim());
    const m = parseInt(max ?? "", 10);
    const t = parseInt(target ?? "", 10);
    if (name && m > 0) out.push({ name, maxTokens: m, targetTokens: t > 0 ? Math.min(t, m) : m });
  }
  return out;
}
var micro = (v) => v === void 0 ? void 0 : Number(v) / 1e6;
function show(r, text) {
  const units = (r.projection?.points ?? []).map((q) => ({ item: q.item, start: q.start, end: q.end, level: q.level, x: q.x, y: q.y, z: q.z, empty: q.empty }));
  const chunks = leaves(units);
  const link = linker();
  const stacks = new Map(r.chunkTermStacks.map((k) => [k.unit, k]));
  const docTerms = r.documentTermStack?.segments ?? [];
  const docTotal = r.documentTermStack?.totalWeight ?? 0;
  const listed2 = spread(chunks.map((c) => stacks.get(c.item)?.segments ?? []));
  const named2 = chunks.map((c) => {
    const k = stacks.get(c.item);
    return k ? distinctive(k.segments, k.totalWeight, docTerms, docTotal, 3, listed2, chunks.length) : [];
  });
  const cosines = new Map((r.projection?.points ?? []).map((q) => [q.item, q.adjacentCosine]));
  const parts = [];
  if (chunks.some((c) => !c.empty)) {
    const model = r.projection?.model;
    const reader = chunkText(text, chunks, link, (term) => {
      const t = docTerms.find((d) => d.term === term);
      return (t?.occStart ?? []).map((a, i) => [a, t?.occEnd?.[i] ?? a]);
    });
    follow(reader, link);
    parts.push(
      el("h3", { text: "Meaning map" }),
      el("p", { class: "muted", text: `Each dot is one chunk's vector from ${model ? `${model.id}@${model.revision}` : "the graph"}, placed by its two main directions of meaning. The line follows reading order: a document that wanders draws a long trail, and one that stays on a topic stays in one place.` }),
      el("div", { class: "map-and-text" }, meaningMap({ units, chunks, variance: r.projection?.explainedVariance ?? [], terms: (i) => named2[i], toNext: (i) => cosines.get(chunks[i].item), text, third: r.projection?.pcaDims === 3 }, link), el("div", {}, el("h4", { text: "The text, by chunk" }), reader))
    );
  } else {
    parts.push(el("h3", { text: "Meaning map" }), el("p", { class: "muted", text: "No vectors in this graph: analyze with a profile that embeds, such as default-v1, and an embedding model." }));
  }
  if (r.driftCurve?.gaps.length) {
    const d = r.driftCurve;
    parts.push(
      el("h3", { text: "Topic seismograph" }),
      seismograph(
        {
          gaps: d.gaps.map((g) => ({ offset: g.gapOffset, raw: micro(g.rawCosineMicro), smoothed: micro(g.smoothedCosineMicro), depth: Number(g.depthScoreMicro) / 1e6, cost: g.cost, reason: g.reason, cut: g.isChunkCut })),
          chunks,
          strategies: (r.strategyMatrix?.strategies ?? []).map((t) => ({ name: t.name, chunks: t.chunkCount, shared: t.matchedCutsWithBaseline, apart: t.disagreementsWithBaseline, cuts: t.cutStarts })),
          text,
          window: d.window,
          smoothing: d.smoothing
        },
        link
      )
    );
  }
  const withStack = chunks.filter((c) => stacks.has(c.item));
  if (docTerms.length && withStack.length > 1) {
    const uneven = input("projection-uneven").checked;
    const terms = riverTerms(docTerms, docTotal, withStack.map((c) => ({ terms: stacks.get(c.item).segments, total: stacks.get(c.item).totalWeight })), 8, uneven);
    const weights = terms.map((t) => withStack.map((c) => stacks.get(c.item).segments.find((x) => x.term === t)?.weight ?? 0));
    if (terms.length) {
      parts.push(
        el("h3", { text: "Term river" }),
        el("p", { class: "muted", text: uneven ? "The terms whose weight is least evenly spread over the chunks, so words every chunk uses alike are left out. A stream is as thick over a chunk as the term's weight there." : "The document's heaviest terms. A stream is as thick over a chunk as the term's weight there." }),
        river({ terms, weights, chunks: withStack, occurrences: (t) => docTerms.find((d) => d.term === t)?.occStart?.length ?? 0, length: text.length }, link)
      );
    }
  }
  const m = r.strategyMatrix;
  const rows2 = chunks.map((c, i) => [String(i + 1), `${c.start}..${c.end}`, named2[i].join(", "), (stacks.get(c.item)?.segments ?? []).slice(0, 6).map((t) => `${t.term} ${t.weight.toFixed(1)}`).join(", "), short(text.slice(c.start, c.end), 60)]);
  parts.push(
    el(
      "details",
      { class: "tables" },
      el("summary", { text: "As tables" }),
      chunks.length ? table(["chunk", "characters", "told apart by", "heaviest terms", "text"], rows2) : "",
      chunks.length ? take("Chunks", () => [["chunk", "start", "end", "told apart by", "x", "y", "text"], ...chunks.map((c, i) => [String(i + 1), String(c.start), String(c.end), named2[i].join(" "), String(c.x), String(c.y), text.slice(c.start, c.end)])]) : "",
      m?.strategies.length ? el("h4", { text: "Chunking budgets" }) : "",
      m?.strategies.length ? table(
        ["strategy", "max", "target", "chunks", "over budget", "cost", "cuts shared", "disagreements"],
        m.strategies.map((t) => [t.name, String(t.budget?.maxTokens ?? ""), String(t.budget?.targetTokens ?? ""), String(t.chunkCount), String(t.overBudgetCount), String(t.totalCost), String(t.matchedCutsWithBaseline), String(t.disagreementsWithBaseline)])
      ) : ""
    )
  );
  box3().replaceChildren(...parts);
}
var freshRender3 = latest();
async function render4() {
  const mine = freshRender3();
  const d = doc();
  if (!d) return void box3().replaceChildren(el("p", { class: "muted", text: "Analyze a document first." }));
  try {
    const r = await analysis.projectChunks({
      graph: d.graph,
      offsets: 1 /* UTF16 */,
      pcaDims: $("projection-dims").value === "3" ? 3 : 2,
      // Enough of each stack to tell a chunk's own terms from the document's.
      maxStackSegments: 32,
      includeDriftCurve: true,
      includeStrategyMatrix: true,
      strategyBudgets: budgets(input("projection-budgets").value)
    });
    if (!mine()) return;
    show(r, d.source);
  } catch (e) {
    if (!mine()) return;
    box3().replaceChildren(failed(explain(e)));
  }
}
function init10() {
  const again = lazy($("analyze-projection"), () => void render4());
  onDoc(() => again());
  const go2 = $("projection-run");
  go2.addEventListener("click", () => void busy(go2, render4));
  $("projection-uneven").addEventListener("change", () => void render4());
}

// demo/web/src/frames.ts
var MAX_FRAME_BYTES = (64 << 20) - 16;
var MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024;
function bytes(b) {
  return b;
}
function prefix(n2) {
  if (n2 > MAX_FRAME_BYTES) throw new RangeError(`a frame of ${n2} bytes is over the ${MAX_FRAME_BYTES} byte limit`);
  const p = new Uint8Array(5);
  new DataView(p.buffer).setUint32(1, n2, false);
  return p;
}
function envelope(message) {
  const out = new Uint8Array(5 + message.length);
  out.set(prefix(message.length));
  out.set(message, 5);
  return out;
}
function varint(n2) {
  const out = [];
  let v = n2;
  while (v >= 128) {
    out.push(v % 128 | 128);
    v = Math.floor(v / 128);
  }
  out.push(v);
  return Uint8Array.from(out);
}
function fieldHead(field3, n2) {
  const tag = varint(field3 * 8 + 2);
  const len = varint(n2);
  const out = new Uint8Array(tag.length + len.length);
  out.set(tag);
  out.set(len, tag.length);
  return out;
}
function textFrame(head, field3, text) {
  const fh = fieldHead(field3, text.size);
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

// demo/web/src/highlight.ts
function boundary(bytes2, at) {
  let i = Math.max(0, Math.min(bytes2.length, at));
  while (i > 0 && i < bytes2.length && (bytes2[i] & 192) === 128) i--;
  return i;
}
function pieces2(text, chunkStart, matches2) {
  const bytes2 = new TextEncoder().encode(text);
  const inside = matches2.map((m) => ({ ...m, start: boundary(bytes2, m.start - chunkStart), end: boundary(bytes2, m.end - chunkStart) })).filter((m, i) => matches2[i].start >= chunkStart && matches2[i].end - chunkStart <= bytes2.length && m.end > m.start && !(m.start === 0 && m.end === bytes2.length));
  const cuts = [.../* @__PURE__ */ new Set([0, bytes2.length, ...inside.flatMap((m) => [m.start, m.end])])].sort((a, b) => a - b);
  const dec = new TextDecoder();
  const out = [];
  for (let i = 0; i + 1 < cuts.length; i++) {
    const [a, b] = [cuts[i], cuts[i + 1]];
    const on = inside.filter((m) => m.start <= a && m.end >= b);
    const piece = { text: dec.decode(bytes2.subarray(a, b)), sources: [...new Set(on.map((m) => m.source))], labels: [...new Set(on.map((m) => m.label).filter(Boolean))] };
    const last2 = out[out.length - 1];
    if (last2 && !last2.sources.length && !piece.sources.length) last2.text += piece.text;
    else out.push(piece);
  }
  return out;
}
function unplaced(text, chunkStart, matches2) {
  const length = new TextEncoder().encode(text).length;
  return matches2.filter((m) => m.start < chunkStart || m.end - chunkStart > length || m.end <= m.start || m.start === chunkStart && m.end - chunkStart === length);
}

// demo/web/src/upload.ts
function put2(method, bodyBlob, signal, sent) {
  return new Promise((resolve, reject) => {
    const stopped = () => reject(new ConnectError("the upload was stopped", Code.Canceled));
    if (signal?.aborted) return stopped();
    const x = new XMLHttpRequest();
    x.open("PUT", `${base()}/upload/${method.parent.typeName}/${method.name}`);
    x.responseType = "arraybuffer";
    x.setRequestHeader("content-type", "application/octet-stream");
    const t = token();
    if (t) x.setRequestHeader("authorization", `Bearer ${t}`);
    const total = bodyBlob.size;
    x.upload.addEventListener("progress", (e) => sent?.(e.loaded, total));
    x.upload.addEventListener("load", () => sent?.(total, total));
    const abort = () => x.abort();
    signal?.addEventListener("abort", abort, { once: true });
    x.addEventListener("loadend", () => signal?.removeEventListener("abort", abort));
    x.addEventListener("abort", stopped);
    x.addEventListener("error", () => reject(new ConnectError("the demo did not answer the upload", Code.Unavailable)));
    x.addEventListener("load", () => {
      if (x.status !== 200) return reject(refusal(x.status, (name) => x.getResponseHeader(name)));
      try {
        resolve(fromBinary(method.output, new Uint8Array(x.response)));
      } catch (e) {
        reject(e);
      }
    });
    sent?.(0, total);
    x.send(bodyBlob);
  });
}
function refusal(http, header) {
  const status3 = Number(header("grpc-status") ?? "");
  const code = status3 > 0 && status3 <= 16 ? status3 : Code.Unknown;
  let message = header("grpc-message") ?? `HTTP ${http}`;
  try {
    message = decodeURIComponent(message);
  } catch {
  }
  if (http === 415) message = "the demo wants application/octet-stream";
  if (http === 413) message = `${message} (HTTP 413: over the 2 GiB cap)`;
  const meta = new Headers();
  for (const k of ["turbo-code", "turbo-field"]) {
    const v = header(k);
    if (v !== null) meta.set(k, v);
  }
  return new ConnectError(message, code, meta);
}

// demo/web/src/tabs/search.ts
var caps = null;
var indexing = null;
var select2 = (id) => $(id);
var freshList2 = latest();
async function list3() {
  const mine = freshList2();
  const box5 = $("search-indexes");
  try {
    const r = await search.listIndexes({});
    if (!mine()) return;
    options(select2("search-index"), r.indexes.map((i) => i.indexName), void 0, "no index yet");
    box5.replaceChildren(
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
          deleteButton2(i.indexName)
        ])
      ) : el("p", { class: "muted", text: "No index yet: create one from an embedding bundle." })
    );
  } catch (e) {
    if (!mine()) return;
    const down = reachable() === false;
    options(select2("search-index"), [], void 0, down ? "the server is not reachable" : "the indexes could not be listed");
    box5.replaceChildren(el("p", { class: "error", text: down ? "The server is not reachable, so its indexes cannot be listed." : describe(e) }));
  }
}
function deleteButton2(name) {
  return confirmed("Delete", `Delete ${name}?`, async () => {
    await search.deleteIndex({ indexName: name });
    await list3();
  }, off(TurboSearchService.method.deleteIndex));
}
async function createIndex(ev) {
  ev.preventDefault();
  const note5 = $("search-create-note");
  const name = input("search-create-name").value.trim();
  const bundle = caps?.bundles.find((b) => b.name === select2("search-create-bundle").value);
  const model = bundle?.model;
  if (!name || !bundle || !model) {
    say(note5, "Name the index and pick an embedding bundle.", "error");
    return;
  }
  await busy(submitter(ev), async () => {
    try {
      await search.createIndex({
        spec: {
          indexName: name,
          space: { model: { id: model.id, revision: model.revision, sha256: model.sha256 }, dim: bundle.widths[0] ?? 0, normalization: 1 /* UNIT */ },
          metric: 0 /* COSINE */,
          storage: 0 /* FLAT_F32 */,
          embedBundle: bundle.name,
          collectionName: input("search-create-collection").value.trim(),
          chunkBudgetTokens: 0
        }
      });
      say(note5, `Created ${name}.`);
      await list3();
      select2("search-index").value = name;
      select2("search-index").dispatchEvent(new Event("change", { bubbles: true }));
    } catch (e) {
      say(note5, explain(e));
    }
  });
}
async function addDocuments() {
  const note5 = $("search-add-note");
  const index = select2("search-index").value;
  const typed = document.getElementById("search-docs").value.split("\n\n").map((t) => t.trim()).filter((t) => t.length);
  const files = [...input("search-files").files ?? []];
  const docs = [
    ...typed.map((t, i) => ({ id: `typed-${Date.now()}-${i + 1}`, text: t })),
    ...files.map((f) => ({ id: f.name, text: f }))
  ];
  if (!index || !docs.length) {
    say(note5, !index ? "Create or pick an index first." : "Type paragraphs (a blank line between) or pick files.", "error");
    return;
  }
  await busy($("search-add"), async () => {
    indexing?.abort();
    const ac = new AbortController();
    indexing = ac;
    try {
      const body2 = indexBody(index, docs);
      const r = await put2(TurboSearchService.method.indexDocuments, body2, ac.signal, sending(note5, count(docs.length, "document")));
      say(note5, `Indexed ${r.indexedDocuments} documents as ${r.indexedChunks} chunks.`);
      await list3();
    } catch (e) {
      say(note5, explain(e));
    } finally {
      if (indexing === ac) indexing = null;
    }
  }, $("search-add-stop"));
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
            (s2) => el("span", {
              class: "hcell",
              title: `${s2.chunkId} ${s2.span?.start ?? 0}..${s2.span?.end ?? 0}: ${s2.score.toFixed(3)}`,
              style: `opacity: ${(0.1 + 0.9 * (d.maxScore ? s2.score / d.maxScore : 0)).toFixed(2)}`
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
  const missing = !select2("search-index").value ? "Pick or create an index first." : !input("search-query").value.trim() ? "Enter something to search for." : "";
  if (missing) return void out.replaceChildren(el("p", { class: "error", text: missing }));
  await busy(submitter(ev), async () => {
    try {
      const r = await search.search({
        indexName: select2("search-index").value,
        naturalQuery: input("search-query").value,
        topK: Math.max(1, parseInt(input("search-k").value, 10) || 10),
        rerankBundle: select2("search-rerank").value
      });
      const sources = [];
      const colour2 = (source) => {
        if (!sources.includes(source)) sources.push(source);
        return `var(--m${sources.indexOf(source) % 8})`;
      };
      const hits = r.hits.map((h, i) => {
        const a = h.attribution;
        const signals = a ? Object.entries(a.signalScores).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(", ") : "";
        const matches2 = (a?.matchedSpans ?? []).map((m) => ({ source: m.source, label: m.label, start: m.span?.start ?? 0, end: m.span?.end ?? 0 }));
        const at = h.chunkSpan?.start ?? 0;
        const text = pieces2(h.text, at, matches2).map(
          (p) => p.sources.length ? el("mark", { style: `--mark: ${colour2(p.sources[0])}`, title: `${p.sources.join(", ")}${p.labels.length ? `: ${p.labels.join(", ")}` : ""}` }, p.text, el("sup", { text: p.sources.join("+") })) : p.text
        );
        const rest = unplaced(h.text, at, matches2).map((m) => `${m.source}${m.label ? `:${m.label}` : ""}`);
        return el(
          "div",
          { class: "hit" },
          el("div", { class: "head" }, el("b", { text: `${i + 1}. ${h.docId}` }), el("span", { class: "tag", text: h.score.toFixed(4) }), el("span", { class: "muted", text: ` ${h.chunkId}` })),
          el("p", { class: "doc flat" }, text),
          signals ? el("p", { class: "muted", text: `signals: ${signals}` }) : null,
          rest.length ? el("p", { class: "muted", text: `matched the whole passage: ${[...new Set(rest)].join("; ")}` }) : null
        );
      });
      const key3 = sources.length ? el("div", { class: "legend" }, el("span", { class: "muted", text: "matched by" }), sources.map((x) => el("span", { class: "swatch", style: `--mark: ${colour2(x)}` }, el("i", {}), ` ${x}`))) : null;
      out.replaceChildren(
        el("p", { class: "muted", text: `${r.hits.length} hits in ${r.resolvedIndex}; query terms: ${r.analyzedQueryTerms.join(" ") || "none"}` }),
        key3 ?? "",
        ...hits,
        r.hits.length ? take(`Search hits ${r.resolvedIndex}`, () => [
          ["rank", "document", "chunk", "score", "text", "signals"],
          ...r.hits.map((h, i) => [String(i + 1), h.docId, h.chunkId, h.score.toFixed(6), h.text, Object.entries(h.attribution?.signalScores ?? {}).map(([k, v]) => `${k} ${v.toFixed(6)}`).join("; ")])
        ]) : "",
        r.documentHeat.length ? el("h3", { text: "Document heat" }) : "",
        r.documentHeat.length ? heatKey("One cell per chunk, in reading order. Stronger colour is a higher score, relative to that document's best (max).") : "",
        heatView(r)
      );
      reveal(out);
    } catch (e) {
      out.replaceChildren(failed(explain(e)));
    }
  });
}
function stop4() {
  indexing?.abort();
  indexing = null;
}
function init11() {
  $("search-create").addEventListener("submit", createIndex);
  lock($("search-create"), TurboSearchService.method.createIndex);
  lock($("search-add-box"), TurboSearchService.method.indexDocuments);
  $("search-add").addEventListener("click", addDocuments);
  $("search-add-stop").addEventListener("click", stop4);
  $("search-form").addEventListener("submit", runSearch);
  const refreshButton = $("search-refresh");
  refreshButton.addEventListener("click", () => void busy(refreshButton, list3));
  const served = () => {
    const embed2 = caps?.bundles.filter((b) => b.task === "TASK_EMBED") ?? [];
    options(select2("search-create-bundle"), embed2.map((b) => b.name), embed2.map((b) => `${b.name} (dim ${b.widths[0] ?? "?"})`), whyNone("embedding model"));
    const rerank2 = caps?.bundles.filter((b) => b.task === "TASK_RERANK").map((b) => b.name) ?? [];
    if (caps) options(select2("search-rerank"), ["", ...rerank2], ["no rerank", ...rerank2]);
    else options(select2("search-rerank"), [], void 0, reachable() === false ? "the server is not reachable" : "no rerank");
  };
  onReachable(() => {
    if (!caps) served();
  });
  onServed(() => void list3());
  onChange(() => void list3());
  onCapabilities((c) => {
    caps = c;
    served();
  });
  onCapabilitiesFailed(served);
}

// demo/web/src/tabs/semantic.ts
var utf8 = new TextEncoder();
var select3 = (id) => $(id);
var caps2 = null;
function param2(p) {
  return create(InferParameterSchema, { parameterChoice: p });
}
function texts(name, values) {
  return { name, datatype: "BYTES", shape: [BigInt(values.length)], contents: { bytesContents: values.map((t) => utf8.encode(t)) } };
}
function output(reply, name) {
  const i = reply.outputs.findIndex((o) => o.name === name);
  const raw = reply.rawOutputContents[i];
  if (i < 0 || !raw) throw new Error(`the reply has no ${name}`);
  return { dv: new DataView(raw.buffer, raw.byteOffset, raw.byteLength), shape: reply.outputs[i].shape.map(Number), datatype: reply.outputs[i].datatype };
}
function role(bundle, r) {
  const b = caps2?.bundles.find((x) => x.name === bundle);
  return b?.roles.includes(r) ? { prompt_role: param2({ case: "stringParam", value: r }) } : {};
}
async function embed(model, query, passages) {
  const out = $("sem-sim");
  const q = await inference.modelInfer({ modelName: model, parameters: role(model, "PROMPT_QUERY"), inputs: [texts("texts", [query])] });
  const p = await inference.modelInfer({ modelName: model, parameters: role(model, "PROMPT_DOCUMENT"), inputs: [texts("texts", passages)] });
  const qv = vectorsOf(q)[0];
  const pv = vectorsOf(p);
  const sim = matrix(pv);
  const k = Math.max(1, parseInt(input("sem-k").value, 10) || 3);
  const near = pv.map((v, i) => ({ i, s: cosine(qv, v) })).sort((a, b) => b.s - a.s);
  out.replaceChildren(
    el("h3", { text: `Nearest neighbours (${model})` }),
    table(["rank", "passage", "cosine"], near.slice(0, k).map((n2, r) => [String(r + 1), el("span", { class: "quote", title: passages[n2.i], text: `#${n2.i + 1} ${short(passages[n2.i], 60)}` }), n2.s.toFixed(4)])),
    take(`Nearest neighbours ${model}`, () => [["rank", "passage number", "passage", "cosine"], ...near.slice(0, k).map((n2, r) => [String(r + 1), String(n2.i + 1), passages[n2.i], n2.s.toFixed(6)])]),
    el("h3", { text: "Passage similarity" }),
    simTable(passages.map((t) => ({ short: short(t, 22), full: t })), sim, `Passage similarity ${model}`)
  );
}
async function rerank(model, query, passages) {
  const meta = await inference.modelMetadata({ name: model });
  const limits = ["model_info.max_batch", "session_info.max_batch"].map((k) => parseInt(meta.properties[k] ?? "", 10)).filter((n2) => n2 > 0);
  const batch = limits.length ? Math.min(...limits) : passages.length;
  const logits = [];
  for (let at = 0; at < passages.length; at += batch) {
    const part2 = passages.slice(at, at + batch);
    const r = await inference.modelInfer({ modelName: model, inputs: [texts("query", [query]), texts("passages", part2)] });
    const { dv } = output(r, "scores");
    part2.forEach((_, i) => logits.push(dv.getFloat32(i * 4, true)));
  }
  const scored = logits.map((s2, i) => ({ i, s: s2 })).sort((a, b) => b.s - a.s);
  $("sem-rerank-out").replaceChildren(
    el("h3", { text: `Reranked (${model})` }),
    el("p", { class: "muted", text: "Relevance logits from the cross-encoder, one per (query, passage) pair; higher is more relevant." }),
    table(["rank", "passage", "logit", "sigmoid"], scored.map((x, r) => [String(r + 1), el("span", { class: "quote", title: passages[x.i], text: `#${x.i + 1} ${short(passages[x.i], 60)}` }), x.s.toFixed(4), (1 / (1 + Math.exp(-x.s))).toFixed(4)])),
    take(`Reranked ${model}`, () => [["rank", "passage number", "passage", "logit", "sigmoid"], ...scored.map((x, r) => [String(r + 1), String(x.i + 1), passages[x.i], x.s.toFixed(6), (1 / (1 + Math.exp(-x.s))).toFixed(6)])])
  );
}
async function answer(model, question, context) {
  const topK = Math.max(1, parseInt(input("sem-topk").value, 10) || 3);
  const r = await inference.modelInfer({
    modelName: model,
    parameters: { top_k: param2({ case: "int64Param", value: BigInt(topK) }) },
    inputs: [texts("question", [question]), texts("context", [context])]
  });
  const spans = output(r, "spans");
  const scores = output(r, "scores");
  const probs = output(r, "probabilities");
  const bytes2 = utf8.encode(context);
  const dec = new TextDecoder();
  const rows2 = [];
  for (let i = 0; i < spans.shape[0]; i++) {
    const [s2, e] = [spans.dv.getInt32(i * 8, true), spans.dv.getInt32(i * 8 + 4, true)];
    rows2.push([String(i + 1), el("span", { class: "quote", text: dec.decode(bytes2.subarray(s2, e)) }), `${s2}..${e}`, scores.dv.getFloat64(i * 8, true).toFixed(3), probs.dv.getFloat64(i * 8, true).toFixed(4)]);
  }
  const p = r.parameters;
  const abstained = p.abstained?.parameterChoice.case === "boolParam" && p.abstained.parameterChoice.value;
  const nullP = p.null_probability?.parameterChoice.case === "doubleParam" ? p.null_probability.parameterChoice.value : NaN;
  $("sem-qa-out").replaceChildren(
    el("h3", { text: `Answers (${model})` }),
    el("p", { class: abstained ? "error" : "muted", text: `${abstained ? "The model abstains: no answer clears the null score." : "Spans in the passages, as UTF-8 byte offsets."}${Number.isNaN(nullP) ? "" : ` No-answer probability ${nullP.toFixed(4)}.`}` }),
    rows2.length ? table(["#", "answer", "bytes", "score", "probability"], rows2) : el("p", { class: "muted", text: "No span." })
  );
}
async function run3() {
  const note5 = $("sem-note");
  const query = input("sem-query").value.trim();
  const passages = $("sem-passages").value.split("\n").map((t) => t.trim()).filter((t) => t.length);
  const [e, rr, qa] = [select3("sem-embed").value, select3("sem-rerank").value, select3("sem-qa").value];
  for (const id of ["sem-sim", "sem-rerank-out", "sem-qa-out"]) $(id).replaceChildren();
  if (!query || !passages.length) return say(note5, "Enter a query and at least one passage.", "error");
  if (!e && !rr && !qa) return say(note5, "Pick at least one model.", "error");
  say(note5, "Running\u2026", "busy");
  const jobs = [
    [e, "sem-sim", () => embed(e, query, passages)],
    [rr, "sem-rerank-out", () => rerank(rr, query, passages)],
    [qa, "sem-qa-out", () => answer(qa, query, passages.join("\n"))]
  ];
  await Promise.all(
    jobs.filter(([m]) => m).map(([, box5, job]) => job().catch((err) => $(box5).replaceChildren(failed(explain(err)))))
  );
  say(note5, "");
  reveal($("sem-sim").parentElement);
}
function init12() {
  const go2 = $("sem-run");
  go2.addEventListener("click", () => void busy(go2, run3));
  const kinds2 = [["sem-embed", "TASK_EMBED", "embedding model"], ["sem-rerank", "TASK_RERANK", "reranker"], ["sem-qa", "TASK_EXTRACTIVE_QA", "QA model"]];
  const served = () => {
    for (const [id, task, what] of kinds2) {
      const names2 = caps2?.bundles.filter((b) => b.task === task).map((b) => b.name) ?? [];
      if (id !== "sem-embed") $(`${id}-pick`).hidden = names2.length === 0;
      if (id === "sem-qa") $("sem-topk-pick").hidden = names2.length === 0;
      if (!names2.length) {
        options(select3(id), [], void 0, whyNone(what));
        continue;
      }
      const first = !select3(id).dataset.filled;
      options(select3(id), ["", ...names2], ["none", ...names2]);
      select3(id).dataset.filled = "yes";
      if (first && !offered(recall(id), ["", ...names2])) select3(id).value = names2[0];
    }
  };
  onReset("semantic", () => {
    for (const [id, task] of kinds2) {
      const first = caps2?.bundles.find((b) => b.task === task)?.name;
      if (first) select3(id).value = first;
    }
  });
  onCapabilities((c) => {
    caps2 = c;
    say($("sem-note"), "");
    served();
  });
  onReachable(() => {
    if (!caps2) served();
  });
  onCapabilitiesFailed((err) => {
    served();
    if (reachable() === false) say($("sem-note"), "");
    else say($("sem-note"), explain(err));
  });
}

// demo/web/src/tabs/server.ts
var EVERY_MS = 1e4;
var checking = null;
function unreached(e) {
  const c = ConnectError.from(e).code;
  return c !== Code.Unauthenticated && c !== Code.PermissionDenied;
}
async function check() {
  const s2 = $("server");
  try {
    const [live, ready, meta] = await Promise.all([inference.serverLive({}), inference.serverReady({}), inference.serverMetadata({})]);
    const state = ready.ready ? "ready" : live.live ? "live, still loading" : "not live";
    s2.textContent = `${meta.name || "server"} ${meta.version}: ${state}`;
    s2.className = ready.ready ? "muted status up" : "muted status";
    setStatus(true, ready.ready);
  } catch (e) {
    if (unreached(e)) {
      s2.textContent = "server not reachable";
      s2.className = "error status down";
      $("down-why").textContent = describe(e);
      setStatus(false, false);
    } else {
      s2.textContent = describe(e);
      s2.className = "error status";
      setStatus(true, false);
    }
  }
}
function status() {
  checking ??= check().finally(() => checking = null);
  return checking;
}
function watch2() {
  setInterval(() => {
    if (!document.hidden) void status();
  }, EVERY_MS);
  addEventListener("focus", () => void status());
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) void status();
  });
  const retry = $("down-retry");
  retry.addEventListener("click", () => void busy(retry, status));
  onReachable((up) => $("down").hidden = up);
  onServed(() => void reload());
}
function renderCapabilities(c) {
  const box5 = $("server-caps");
  const bundles2 = table(
    ["name", "task", "model", "widths", "context", "batch", "roles", ""],
    c.bundles.map((b) => [
      b.name,
      plain2(b.task, "TASK_"),
      b.model ? `${b.model.id}@${b.model.revision}` : "",
      b.widths.join(", "),
      String(b.contextLimit),
      String(b.batchLimit),
      b.roles.map((r) => plain2(r, "PROMPT_")).join(", "),
      unloadButton(b.name)
    ])
  );
  const profiles = table(
    ["profile", "version", "steps", "layers", "needs"],
    c.profiles.map((p) => [p.name, p.version, p.steps.join(" \u2192 "), p.layers.join(", "), p.requiredBundles.join(", ")])
  );
  const l = c.server;
  const limits = l ? el("p", { class: "muted", text: `At most ${l.maxLoadedModels} served models and ${l.maxSessionsPerModel} sessions per model.` }) : "";
  box5.replaceChildren(el("h3", { text: "Bundles" }), bundles2, limits, el("h3", { text: "Analysis profiles" }), profiles);
}
function unloadButton(name) {
  return confirmed("Unload", `Unload ${name}?`, async () => {
    await analysis.unloadBundle({ name });
    say($("server-load-note"), `Unloaded ${name}.`);
    await reload();
  }, off(AnalysisService.method.unloadBundle));
}
async function reload() {
  try {
    renderCapabilities(await refresh());
  } catch (e) {
    const why = reachable() === false ? "The server is not reachable, so what it serves cannot be listed." : describe(e);
    $("server-caps").replaceChildren(el("p", { class: "error", text: why }));
  }
}
async function load2(ev) {
  ev.preventDefault();
  const note5 = $("server-load-note");
  if (!input("server-load-name").value.trim() || !input("server-load-path").value.trim()) return say(note5, "Name the bundle and give its path on the server.", "error");
  say(note5, "Loading\u2026", "busy");
  await busy(submitter(ev), async () => {
    try {
      const r = await analysis.loadBundle({
        name: input("server-load-name").value.trim(),
        bundle: input("server-load-path").value.trim(),
        sessions: Math.max(1, parseInt(input("server-load-sessions").value, 10) || 1),
        selectDevice: input("server-load-select").checked
      });
      say(note5, `Loaded ${r.capabilities?.name ?? ""} (${r.capabilities?.task ?? ""}).`);
      await reload();
    } catch (e) {
      say(note5, explain(e));
    }
  });
}
function init13() {
  const t = input("token");
  t.value = token();
  t.addEventListener("change", () => {
    setToken(t.value.trim());
    void status();
    void permits().then(reload);
  });
  $("server-load").addEventListener("submit", load2);
  lock($("server-load"), AnalysisService.method.loadBundle);
  watch2();
  void status();
}

// demo/web/src/tabs/vocabulary.ts
var inFlight = /* @__PURE__ */ new Set();
var select4 = (id) => $(id);
async function stoppable2(job) {
  const ac = new AbortController();
  inFlight.add(ac);
  try {
    await job(ac.signal);
  } finally {
    inFlight.delete(ac);
  }
}
function stop5() {
  for (const ac of inFlight) ac.abort();
  inFlight.clear();
}
var stopButton = () => $("vocab-stop");
function actions(id) {
  const get = el("button", { type: "button", text: "Download" });
  get.addEventListener("click", () => void busy(get, () => stoppable2((signal) => fetchArtifact(id, signal)), stopButton()));
  const del = confirmed("Delete", `Delete ${id}?`, async () => {
    await vocabulary.deleteVocabularyArtifact({ artifactId: id });
    await list4();
  }, off(TurboVocabularyService.method.deleteVocabularyArtifact));
  return el("span", {}, get, " ", del);
}
var freshList3 = latest();
async function list4() {
  const mine = freshList3();
  const box5 = $("vocab-lists");
  try {
    const [f, d, v] = await Promise.all([vocabulary.listDictionaryFormats({}), vocabulary.listDictionaries({}), vocabulary.listVocabularies({})]);
    if (!mine()) return;
    const importable = f.formats.filter((x) => x.importable);
    options(select4("vocab-format"), importable.map((x) => x.formatId), importable.map((x) => `${x.formatId}: ${x.description}`), "the server imports no format");
    options(select4("vocab-seed"), ["", ...d.dictionaries.map((x) => x.artifactId)], ["no seed dictionary", ...d.dictionaries.map((x) => x.name || x.artifactId)]);
    const l = f.limits;
    box5.replaceChildren(
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
    if (!mine()) return;
    const down = reachable() === false;
    for (const id of ["vocab-format", "vocab-seed"]) options(select4(id), [], void 0, down ? "the server is not reachable" : "the store could not be listed");
    box5.replaceChildren(el("p", { class: "error", text: down ? "The server is not reachable, so its store cannot be listed." : describe(e) }));
  }
}
async function importDictionary(ev) {
  ev.preventDefault();
  const note5 = $("vocab-note");
  const file = input("vocab-file").files?.[0];
  if (!file) {
    say(note5, "Pick a dictionary file.", "error");
    return;
  }
  await busy(submitter(ev), () => stoppable2(async (signal) => {
    try {
      const sha = await sha256Blob(file, (done) => say(note5, `Hashing ${size(done)} of ${size(file.size)}\u2026`, "busy"));
      const d = await put2(
        TurboVocabularyService.method.importDictionary,
        importBody(
          {
            formatId: select4("vocab-format").value,
            language: input("vocab-language").value.trim(),
            name: input("vocab-name").value.trim() || file.name,
            byteSize: BigInt(file.size),
            sha256: sha,
            hasHeaderRow: input("vocab-header").checked
          },
          file
        ),
        signal,
        sending(note5, file.name)
      );
      say(note5, `Imported ${d.name || d.artifactId}: ${d.entryCount} entries.`);
      await list4();
    } catch (e) {
      say(note5, explain(e));
    }
  }), stopButton());
}
async function learn(ev) {
  ev.preventDefault();
  const note5 = $("vocab-note");
  const typed = document.getElementById("vocab-docs").value.split("\n\n").map((t) => t.trim()).filter((t) => t.length);
  const docs = [...typed, ...[...input("vocab-learn-files").files ?? []]];
  if (!docs.length) {
    say(note5, "Type documents (a blank line between) or pick files.", "error");
    return;
  }
  const language = input("vocab-learn-language").value.trim();
  if (input("vocab-stopwords").checked && !language) {
    say(note5, "Dropping stopwords needs a language, such as en: the server drops that language's bundled list.", "error");
    return;
  }
  await busy(submitter(ev), () => stoppable2(async (signal) => {
    try {
      const body2 = learnBody(
        {
          language,
          minTermFrequency: parseInt(input("vocab-min-tf").value, 10) || 1,
          maxTerms: parseInt(input("vocab-max-terms").value, 10) || 0,
          lowercase: input("vocab-lowercase").checked,
          filterStopwords: input("vocab-stopwords").checked,
          includeNgrams: input("vocab-ngrams").checked,
          minNgramSize: 2,
          maxNgramSize: 3,
          dictionaryArtifactId: select4("vocab-seed").value
        },
        docs
      );
      const v = await put2(TurboVocabularyService.method.learnVocabulary, body2, signal, sending(note5, count(docs.length, "document")));
      say(note5, `Learned ${v.artifactId}: ${v.termCount} terms and ${v.ngramCount} n-grams from ${v.documents} documents.`);
      await list4();
    } catch (e) {
      say(note5, explain(e));
    }
  }), stopButton());
}
async function fetchArtifact(id, signal) {
  const note5 = $("vocab-note");
  const parts = [];
  let name = id;
  let got = 0;
  try {
    for await (const c of vocabulary.downloadVocabulary({ artifactId: id }, { signal })) {
      const f = c.frame;
      if (f.case === "dictionary") name = `${f.value.name || id}.dictionary`;
      else if (f.case === "vocabulary") name = `${id}.vocabulary`;
      else if (f.case === "data") {
        parts.push(f.value);
        got += f.value.length;
        say(note5, `Downloading ${name}: ${size(got)}\u2026`, "busy");
      }
    }
    download(name, parts, "application/octet-stream");
    say(note5, `Downloaded ${name}, ${size(got)}.`);
  } catch (e) {
    say(note5, explain(e));
  }
}
function init14() {
  $("vocab-import").addEventListener("submit", importDictionary);
  $("vocab-learn").addEventListener("submit", learn);
  lock($("vocab-import"), TurboVocabularyService.method.importDictionary);
  lock($("vocab-learn"), TurboVocabularyService.method.learnVocabulary);
  const again = $("vocab-refresh");
  again.addEventListener("click", () => void busy(again, list4));
  onServed(() => void list4());
  onChange(() => void list4());
  $("vocab-stop").addEventListener("click", stop5);
}

// demo/web/src/graphwire.ts
var MAGIC = "TGRF";
var FORMAT_VERSION = 3;
var LISTED2 = 64;
var WireError = class extends Error {
  constructor(message, at) {
    super(`${message} at byte ${at}`);
    this.at = at;
  }
  at;
};
var Cursor = class {
  constructor(buf) {
    this.buf = buf;
    this.dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  buf;
  at = 0;
  dv;
  utf8 = new TextDecoder("utf-8", { fatal: true });
  take(n2) {
    if (this.at + n2 > this.buf.length) throw new WireError("truncated", this.at);
    const out = this.buf.subarray(this.at, this.at + n2);
    this.at += n2;
    return out;
  }
  u8() {
    return this.take(1)[0];
  }
  u16() {
    const v = this.dv.getUint16(this.at, true);
    this.take(2);
    return v;
  }
  u32() {
    if (this.at + 4 > this.buf.length) throw new WireError("truncated", this.at);
    const v = this.dv.getUint32(this.at, true);
    this.at += 4;
    return v;
  }
  u64() {
    if (this.at + 8 > this.buf.length) throw new WireError("truncated", this.at);
    const v = this.dv.getBigUint64(this.at, true);
    this.at += 8;
    return v;
  }
  str() {
    const start2 = this.at;
    const b = this.take(this.u32());
    try {
      return this.utf8.decode(b);
    } catch {
      throw new WireError("a string that is not UTF-8", start2);
    }
  }
};
function part(c, name, read) {
  const at = c.at;
  const r = read(c);
  return { name, at, len: c.at - at, ...r };
}
function str(c, name) {
  return part(c, name, (c2) => ({ value: c2.str() }));
}
function tagError(c, what, tag) {
  return new WireError(`${what} tag of ${tag}`, c.at - 1);
}
function layerRef(c, name) {
  return part(c, name, (c2) => ({ value: `${c2.str()}#${c2.str()}` }));
}
function edits(c, name) {
  return part(c, name, (c2) => {
    const n2 = c2.u32();
    const parts = [];
    for (let i = 0; i < n2; i++) {
      const at = c2.at;
      const [ps, pe, cs, ce] = [c2.u32(), c2.u32(), c2.u32(), c2.u32()];
      if (i < LISTED2) parts.push({ name: `edit ${i}`, at, len: 16, value: `${ps}..${pe} \u2192 ${cs}..${ce}` });
    }
    return { value: `${n2} edits${n2 > LISTED2 ? `, first ${LISTED2} listed` : ""}`, parts };
  });
}
function error(c) {
  return part(c, "error", (c2) => {
    const tag = c2.u8();
    const one = (kind, parts) => ({ value: kind, parts });
    switch (tag) {
      case 0:
        return one("missing", [layerRef(c2, "layer")]);
      case 1:
        return one("duplicate", [layerRef(c2, "layer")]);
      case 2:
        return one("clash", [layerRef(c2, "layer")]);
      case 3:
        return one("bounds", [layerRef(c2, "layer"), str(c2, "view"), str(c2, "detail")]);
      case 4:
        return one("misaligned", [layerRef(c2, "layer"), layerRef(c2, "anchor"), u32(c2, "len"), u32(c2, "anchor len")]);
      case 5:
        return one("unaligned", [layerRef(c2, "layer"), layerRef(c2, "on"), str(c2, "read")]);
      case 6:
        return one("invalid", [layerRef(c2, "layer"), str(c2, "detail")]);
      case 7:
        return one("pipeline", [str(c2, "detail")]);
      case 8:
        return one("annotator", [str(c2, "annotator"), str(c2, "detail")]);
      case 9:
        return one("skipped", [str(c2, "reason")]);
      case 10:
        return one("offset", [
          part(c2, "unit", (c3) => {
            const u = c3.u8();
            if (u > 1) throw tagError(c3, "an offset unit", u);
            return { value: u === 0 ? "utf16" : "code point" };
          }),
          u32(c2, "at"),
          part(c2, "reason", (c3) => {
            const r = c3.u8();
            if (r === 0) return { value: `past end, len ${c3.u32()}` };
            if (r === 1) return { value: "inside a character" };
            throw tagError(c3, "an offset reason", r);
          })
        ]);
      case 11:
        return one("source", [
          part(c2, "reason", (c3) => {
            const r = c3.u8();
            if (r === 0) return { value: `too long, ${c3.u64()} bytes` };
            if (r === 1) return { value: `misaligned: ${c3.str()}` };
            throw tagError(c3, "a source error", r);
          })
        ]);
      case 12:
        return one("over limit", [str(c2, "annotator"), str(c2, "limit"), u32(c2, "max"), u32(c2, "got")]);
      default:
        throw tagError(c2, "an error", tag);
    }
  });
}
function u32(c, name) {
  return part(c, name, (c2) => ({ value: String(c2.u32()) }));
}
function status2(c) {
  return part(c, "status", (c2) => {
    const tag = c2.u8();
    if (tag === 0) return { value: "complete" };
    if (tag === 1) return { value: `skipped: ${c2.str()}` };
    if (tag === 2) return { value: "failed", parts: [error(c2)] };
    throw tagError(c2, "a status", tag);
  });
}
function option(c, name, what, read) {
  return part(c, name, (c2) => {
    const tag = c2.u8();
    if (tag === 0) return { value: "none" };
    if (tag === 1) return read(c2);
    throw tagError(c2, what, tag);
  });
}
function refs(c, name) {
  return part(c, name, (c2) => {
    const n2 = c2.u32();
    const parts = [];
    for (let i = 0; i < n2; i++) parts.push(layerRef(c2, `${name} ${i}`));
    return { value: parts.map((p) => p.value).join(", ") || "none", parts };
  });
}
function digest(c, name) {
  return option(c, name, "a digest", (c2) => {
    const b = c2.take(32);
    let s2 = "";
    for (const x of b) s2 += x.toString(16).padStart(2, "0");
    return { value: s2 };
  });
}
function provenance(c) {
  return part(c, "provenance", (c2) => {
    let hasDigests = false;
    const parts = [
      str(c2, "annotator"),
      str(c2, "version"),
      part(c2, "config", (c3) => ({ value: c3.u64().toString(16).padStart(16, "0") })),
      option(c2, "model", "a model", (c3) => ({ parts: [str(c3, "id"), str(c3, "revision"), str(c3, "sha256")] })),
      part(c2, "execution", (c3) => {
        const raw = c3.u8();
        hasDigests = (raw & 128) !== 0;
        const tag = raw & 127;
        if (tag === 0) return { value: "none" };
        if (tag === 1) return { parts: [str(c3, "backend"), str(c3, "device"), str(c3, "precision")] };
        throw tagError(c3, "an execution", raw);
      }),
      refs(c2, "reads"),
      refs(c2, "on")
    ];
    if (hasDigests) {
      parts.push(digest(c2, "provenance digest"), digest(c2, "input snapshot"));
    }
    return { value: `${parts[0].value} ${parts[1].value}`, parts };
  });
}
function sized(c, name) {
  return part(c, name, (c2) => {
    const n2 = c2.u32();
    c2.take(n2);
    return { value: `${n2} bytes` };
  });
}
function walkGraph(bytes2) {
  const c = new Cursor(bytes2);
  const parts = [];
  parts.push(
    part(c, "magic", (c2) => {
      const m = String.fromCharCode(...c2.take(4));
      if (m !== MAGIC) throw new WireError("not a document graph (bad magic)", 0);
      return { value: m };
    })
  );
  parts.push(
    part(c, "format version", (c2) => {
      const v = c2.u16();
      if (v !== FORMAT_VERSION) throw new WireError(`format version ${v}, not ${FORMAT_VERSION}`, 4);
      return { value: String(v) };
    })
  );
  parts.push(sized(c, "source bytes"));
  parts.push(part(c, "source text", (c2) => ({ value: `${c2.str().length} UTF-16 units` })));
  parts.push(option(c, "source alignment", "an alignment", (c2) => ({ parts: [edits(c2, "edits")] })));
  parts.push(
    part(c, "views", (c2) => {
      const n2 = c2.u32();
      const views = [];
      for (let i = 0; i < n2; i++) {
        views.push(
          part(c2, `view ${i}`, (c3) => {
            const p = [
              str(c3, "instance"),
              status2(c3),
              provenance(c3),
              option(c3, "parent", "a parent", (c4) => ({ value: c4.str() })),
              option(c3, "text", "a text", (c4) => ({ value: `${c4.str().length} UTF-16 units` })),
              edits(c3, "edits")
            ];
            return { value: p[0].value, parts: p };
          })
        );
      }
      return { value: `${n2} views`, parts: views };
    })
  );
  parts.push(
    part(c, "layers", (c2) => {
      const n2 = c2.u32();
      const layers = [];
      for (let i = 0; i < n2; i++) {
        layers.push(
          part(c2, `layer ${i}`, (c3) => {
            const p = [
              str(c3, "layer"),
              str(c3, "instance"),
              part(c3, "schema", (c4) => ({ value: String(c4.u16()) })),
              u32(c3, "items"),
              status2(c3),
              provenance(c3),
              option(c3, "body", "a body", (c4) => {
                const n3 = c4.u32();
                c4.take(n3);
                return { value: `${n3} bytes` };
              })
            ];
            return { value: `${p[0].value}#${p[1].value}`, parts: p };
          })
        );
      }
      return { value: `${n2} layers`, parts: layers };
    })
  );
  if (c.at !== bytes2.length) throw new WireError(`${bytes2.length - c.at} trailing bytes`, c.at);
  return { name: "document graph", at: 0, len: bytes2.length, value: `format ${FORMAT_VERSION}`, parts };
}
function hex(bytes2, p, n2 = 32) {
  const b = bytes2.subarray(p.at, p.at + Math.min(p.len, n2));
  let s2 = "";
  for (const x of b) s2 += (s2 ? " " : "") + x.toString(16).padStart(2, "0");
  return p.len > n2 ? `${s2} \u2026` : s2;
}

// demo/web/src/tabs/wire.ts
var box4 = () => $("wire-out");
var note4 = () => $("wire-note");
function bar2(p, total) {
  const pct = total ? 100 * p.len / total : 0;
  return el("span", { class: "share", title: `${pct.toFixed(1)}%` }, el("span", { style: `width: ${pct.toFixed(2)}%` }));
}
function tree(bytes2, p, total) {
  const line = el(
    "span",
    { class: "wire", title: `bytes ${p.at}..${p.at + p.len}: ${hex(bytes2, p)}` },
    el("b", { text: p.name }),
    p.value ? el("span", { class: "muted", text: ` ${p.value}` }) : "",
    el("span", { class: "size", text: size(p.len) }),
    bar2(p, total)
  );
  if (!p.parts?.length) return el("div", { class: "leaf" }, line);
  return el("details", {}, el("summary", {}, line), el("div", { class: "kids" }, p.parts.map((c) => tree(bytes2, c, total))));
}
function render5() {
  const d = doc();
  if (!d) return void box4().replaceChildren(el("p", { class: "muted", text: "Analyze a document or open a graph file." }));
  try {
    const g = walkGraph(d.graph);
    const layers = g.parts?.find((p) => p.name === "layers")?.parts ?? [];
    const bodies = layers.reduce((n2, l) => n2 + (l.parts?.find((p) => p.name === "body")?.len ?? 0), 0);
    box4().replaceChildren(
      el("p", { class: "muted", text: `${size(d.graph.length)} in all, ${size(bodies)} of it layer bodies. Open a part for its fields; hover for its offsets and first bytes.` }),
      el("div", { class: "kids top" }, (g.parts ?? []).map((p) => tree(d.graph, p, g.len)))
    );
  } catch (e) {
    const where = e instanceof WireError ? ` (byte ${e.at})` : "";
    box4().replaceChildren(el("p", { class: "error", text: `${e instanceof Error ? e.message : String(e)}${where}` }));
  }
}
async function roundTrip() {
  const d = doc();
  if (!d) return say(note4(), "Analyze a document or open a graph file.", "error");
  try {
    const r = await analysis.roundTripDocumentGraph({ graph: d.graph });
    const same = r.graph.length === d.graph.length && r.graph.every((b, i) => b === d.graph[i]);
    say(note4(), `The server re-exported ${size(r.graph.length)}, ${r.layers.length} layers and ${r.views.length} views: ${same ? "byte for byte the same" : "different bytes"}.`, same ? "" : "error");
  } catch (e) {
    say(note4(), explain(e));
  }
}
async function open() {
  const f = input("wire-file").files?.[0];
  if (!f) return;
  try {
    const graph = new Uint8Array(await f.arrayBuffer());
    const r = await analysis.roundTripDocumentGraph({ graph });
    setDoc({ graph, views: r.views, source: r.sourceText });
    say(note4(), `Opened ${f.name}: ${r.layers.length} layers, ${r.views.length} views.`);
  } catch (e) {
    say(note4(), explain(e));
  }
}
function init15() {
  const again = lazy($("analyze-wire"), render5);
  onDoc(() => again());
  const go2 = $("wire-roundtrip");
  go2.addEventListener("click", () => void busy(go2, roundTrip));
  $("wire-file").addEventListener("change", () => void open());
  $("wire-save").addEventListener("click", () => {
    const d = doc();
    if (d) download("document.graph", [d.graph], "application/octet-stream");
  });
}

// demo/web/src/main.ts
var stops = {
  analyze: stop,
  search: () => (stop4(), stop2()),
  vocabulary: stop5,
  models: stop3
};
function show2(name) {
  for (const [tab, stop6] of Object.entries(stops)) if (tab !== name) stop6();
  for (const b of document.querySelectorAll("nav.tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === name));
  for (const p of document.querySelectorAll("section.panel")) p.hidden = p.id !== `tab-${name}`;
}
function named() {
  const want = location.hash.slice(1);
  return document.getElementById(`tab-${want}`) ? want : "embed";
}
function go(name) {
  if (location.hash !== `#${name}`) history.pushState(null, "", `#${name}`);
  show2(name);
}
async function start() {
  const tabs = [...document.querySelectorAll("nav.tabs button")];
  tabs.forEach((b, i) => {
    b.addEventListener("click", () => go(b.dataset.tab ?? "embed"));
    b.addEventListener("keydown", (ev) => {
      const step = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
      if (!step) return;
      ev.preventDefault();
      const next = tabs[(i + step + tabs.length) % tabs.length];
      next.focus();
      go(next.dataset.tab ?? "embed");
    });
  });
  await permits();
  init();
  init7();
  init13();
  init6();
  init12();
  init2();
  init4();
  init5();
  init10();
  init15();
  init11();
  init3();
  init14();
  init9();
  addEventListener("hashchange", () => show2(named()));
  addEventListener("popstate", () => show2(named()));
  const first = named();
  if (location.hash !== `#${first}`) history.replaceState(null, "", `#${first}`);
  show2(first);
}
void start();
