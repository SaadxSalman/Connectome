# BSON Format Guide

## Introduction to BSON
**BSON** (Binary JSON) is a binary format used to store documents composed of key/value pairs. It provides a flexible, efficient way to represent structured data and is primarily used in MongoDB. Each BSON document contains ordered key/value pairs, serialized in a compact format for storage and transmission.

---

## BSON Grammar (Version 1.1)
The BSON format follows a specific grammar defined using pseudo-BNF (Backus-Naur Form) syntax. The core unit of BSON data is a **document**.

### Grammar Structure
- **document**: A BSON document is a sequence of key/value pairs, prefixed by its total length in bytes and ending with a null byte.
- **e_list**: Represents a list of elements in the document.
- **element**: Represents a key/value pair within the document.

Below is the grammar structure in BNF notation:
```plaintext
document ::= int32 e_list unsigned_byte(0)
e_list  ::= element e_list | ""
element ::= signed_byte type_code e_name value
```

---

## Basic Types
BSON uses various basic data types for representing values within a document. These types are serialized in **little-endian format**.

| Type          | Description                                         | Size       |
|---------------|-----------------------------------------------------|------------|
| byte          | 8-bit data                                          | 1 byte     |
| signed_byte(n)| 8-bit signed integer with value `n`                 | 1 byte     |
| int32         | 32-bit signed integer                               | 4 bytes    |
| int64         | 64-bit signed integer                               | 8 bytes    |
| uint64        | 64-bit unsigned integer                             | 8 bytes    |
| double        | 64-bit IEEE floating-point number                   | 8 bytes    |
| decimal128    | 128-bit decimal floating-point number               | 16 bytes   |

---

## Non-Terminals
The following non-terminal elements define the rest of the BSON grammar:

### Document Structure
- **document**: Starts with a 32-bit integer indicating the total byte size, followed by a list of elements and a null byte terminator.
- **e_list**: A recursive structure representing multiple elements in a document.
- **element**: Represents a key/value pair, identified by a type code and key name, followed by the value.

### Example:
```plaintext
document ::= int32 e_list unsigned_byte(0)
e_list ::= element e_list | ""
```

### Value Types
Each element's value is determined by its type code. Below is a breakdown of type codes and their corresponding data types:

| Type Code | Description                             | Value Representation |
|-----------|-----------------------------------------|----------------------|
| 1         | 64-bit floating point                   | double               |
| 2         | UTF-8 string                            | string               |
| 3         | Embedded document                      | document             |
| 4         | Array                                   | document (array)     |
| 5         | Binary data                             | binary               |
| 6         | Undefined (deprecated)                  |                      |
| 7         | ObjectId                                | 12-byte identifier   |
| 8         | Boolean                                 | unsigned_byte(0/1)   |
| 9         | UTC datetime                            | int64                |
| 10        | Null value                              |                      |
| 11        | Regular expression                     | cstring cstring      |
| 12        | DBPointer (deprecated)                  | string (byte*12)     |
| 13        | JavaScript code                         | string               |
| 14        | Symbol (deprecated)                     | string               |
| 15        | JavaScript code with scope (deprecated) | code_w_s             |
| 16        | 32-bit integer                          | int32                |
| 17        | Timestamp                               | uint64               |
| 18        | 64-bit integer                          | int64                |
| 19        | 128-bit decimal floating point          | decimal128           |
| -1        | Min key                                 |                      |
| 127       | Max key                                 |                      |

---

## Strings and Binary Data
### Strings
A **string** in BSON is serialized as follows:
```plaintext
string ::= int32 (byte*) unsigned_byte(0)
```
- **int32**: Total length of the string in bytes, including the null terminator.
- **byte***: UTF-8 encoded characters.
- **unsigned_byte(0)**: Null byte terminator.

### cstring
A **cstring** is a modified UTF-8 string that must not contain null bytes within its content.
```plaintext
cstring ::= (byte*) unsigned_byte(0)
```

### Binary Data
Binary data is represented using the **binary** non-terminal:
```plaintext
binary ::= int32 subtype (byte*)
```
- **int32**: Length of the binary data.
- **subtype**: Specifies the binary data's type.
- **byte***: The binary content.

### Binary Subtypes
| Subtype Code | Description                       |
|--------------|-----------------------------------|
| 0            | Generic binary subtype            |
| 1            | Function                          |
| 2            | Binary (Old, deprecated)          |
| 3            | UUID (Old, deprecated)            |
| 4            | UUID                              |
| 5            | MD5                               |
| 6            | Encrypted BSON value              |
| 7            | Compressed BSON column            |
| 8            | Sensitive                         |
| 9            | Vector                            |
| 128-255      | User-defined                      |

---

## Special Types and Notes
- **Array**: BSON arrays are serialized as documents with integer keys (e.g., {'0': 'red', '1': 'blue'}).
- **UTC datetime**: Stored as the number of milliseconds since the Unix epoch.
- **Timestamp**: Used internally by MongoDB for replication and sharding.
- **Min Key / Max Key**: Special values used for comparing BSON documents.

---

## Summary
BSON is a highly efficient, binary-encoded format designed for storing and querying structured data. It supports a variety of data types and provides flexibility for representing complex documents. MongoDB uses BSON as its native data format, making it an essential part of its ecosystem.

