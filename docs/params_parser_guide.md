---
title: UltraDMRG Params Parser Guide
---

# UltraDMRG Params Parser Guide

This guide documents the usage of `qlmps::CaseParamsParserBasic` for reading
simulation parameters from a JSON file.

## JSON Structure

Parameters live under the top-level object `CaseParams`:

```json
{
  "CaseParams": {
    "Int": 1,
    "Double": 2.33,
    "Char": "c",
    "String": "string",
    "Boolean": false,
    "IntVec": [1, -1, 3],
    "SizeTVec": [4, 7, 10],
    "DoubleVec": [0.1, 0.02, 0.003]
  }
}
```

## Construction

```cpp
qlmps::CaseParamsParserBasic parser(argv[1]);
```

If the file cannot be opened or `CaseParams` is missing, the constructor prints a
clear error (including file path) and exits with code 1.

## Strict Parse APIs (exit on failure)

```cpp
int           ParseInt(const std::string& key);
double        ParseDouble(const std::string& key);
char          ParseChar(const std::string& key);     // empty string -> '\0'
std::string   ParseStr(const std::string& key);
bool          ParseBool(const std::string& key);
std::vector<int>    ParseIntVec(const std::string& key);
std::vector<size_t> ParseSizeTVec(const std::string& key);
std::vector<double> ParseDoubleVec(const std::string& key);
```

- On missing key or wrong type, these functions print a detailed error message
  (file path, key, expected type, actual type) and exit with code 1.
- `ParseChar` returns `'\0'` if the string is empty.

## Parse With Default (only when key is missing)

```cpp
int                 ParseIntOr(const std::string& key, int defv);
double              ParseDoubleOr(const std::string& key, double defv);
char                ParseCharOr(const std::string& key, char defv);
std::string         ParseStrOr(const std::string& key, const std::string& defv);
bool                ParseBoolOr(const std::string& key, bool defv);
std::vector<int>    ParseIntVecOr(const std::string& key, const std::vector<int>& defv);
std::vector<size_t> ParseSizeTVecOr(const std::string& key, const std::vector<size_t>& defv);
std::vector<double> ParseDoubleVecOr(const std::string& key, const std::vector<double>& defv);
```

- If the key is missing, return the default value.
- If the key exists but the type is wrong, behavior matches strict parse (print
  error and exit). This prevents silently accepting wrong types.

## TryParse APIs (non-fatal)

```cpp
bool TryParseInt(const std::string& key, int& out);
bool TryParseDouble(const std::string& key, double& out);
bool TryParseChar(const std::string& key, char& out);     // empty string -> out='\0'
bool TryParseStr(const std::string& key, std::string& out);
bool TryParseBool(const std::string& key, bool& out);
bool TryParseIntVec(const std::string& key, std::vector<int>& out);
bool TryParseSizeTVec(const std::string& key, std::vector<size_t>& out);
bool TryParseDoubleVec(const std::string& key, std::vector<double>& out);
```

- Return `true` on success.
- Return `false` if key is missing or value type is invalid.
- Never exit the program.

## Existence Check

```cpp
bool Has(const std::string& key) const;
```

## Error Messages

Strict parse (and default parse when key exists) produce clear diagnostics, e.g.:

```
CaseParams parse error in file '/path/to/input.json':
  key 'Double' has invalid type
    expected: number
    actual:   string
```

## Best Practices

- Use strict parse when the parameter is mandatory.
- Use `Parse*Or` for optional parameters with sensible defaults.
- Use `TryParse*` when you want to handle absence or type errors gracefully.
- Prefer early validation to catch user errors sooner.


