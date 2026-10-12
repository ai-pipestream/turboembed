package ai.pipestream.turbo.goldens;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/** SHA-256 and SHA-512 of a byte array as lowercase hex, through the JDK's MessageDigest. */
final class Sha {

  private Sha() {}

  static String sha256(byte[] bytes) {
    return hex(digest("SHA-256", bytes));
  }

  static String sha512(byte[] bytes) {
    return hex(digest("SHA-512", bytes));
  }

  private static byte[] digest(String algorithm, byte[] bytes) {
    try {
      return MessageDigest.getInstance(algorithm).digest(bytes);
    } catch (NoSuchAlgorithmException e) {
      // Every Java platform is required to provide both algorithms.
      throw new IllegalStateException(algorithm, e);
    }
  }

  static String hex(byte[] bytes) {
    char[] digits = "0123456789abcdef".toCharArray();
    char[] out = new char[bytes.length * 2];
    for (int i = 0; i < bytes.length; i++) {
      out[2 * i] = digits[(bytes[i] >> 4) & 0xf];
      out[2 * i + 1] = digits[bytes[i] & 0xf];
    }
    return new String(out);
  }
}
