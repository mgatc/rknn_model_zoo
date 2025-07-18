// Copyright (C) 2011  Carl Rogers
// Released under MIT License
// license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

// #include "Logging.h"

#include <stdint.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace cnpy {

struct NpyArray
{
  NpyArray(const std::vector<size_t>& _shape, size_t _word_size, bool _fortran_order, std::string _typeName)
    : shape(_shape)
    , word_size(_word_size)
    , fortran_order(_fortran_order)
    , typeName(_typeName)
  {
    num_vals = 1;
    for (size_t i = 0; i < shape.size(); i++)
      num_vals *= shape[i];
    data_holder = std::shared_ptr<std::vector<char>>(new std::vector<char>(num_vals * word_size));
  }

  NpyArray()
    : shape(0)
    , word_size(0)
    , fortran_order(0)
    , num_vals(0)
  {}

  template <typename T>
  T* data()
  {
    return reinterpret_cast<T*>(&(*data_holder)[0]);
  }

  template <typename T>
  const T* data() const
  {
    return reinterpret_cast<T*>(&(*data_holder)[0]);
  }

  template <typename T>
  std::vector<T> as_vec() const
  {
    const T* p = data<T>();
    return std::vector<T>(p, p + num_vals);
  }

  size_t num_bytes() const { return data_holder->size(); }

  std::shared_ptr<std::vector<char>> data_holder;
  std::vector<size_t>                shape;
  size_t                             word_size;
  bool                               fortran_order;
  size_t                             num_vals;
  std::string                        typeName;
};


char BigEndianTest(int size);
char map_type(const std::type_info& t);
template <typename T>
std::vector<char> create_npy_header(const std::vector<size_t>& shape);
void              parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order,
                                   std::string& typeName);
void     parse_npy_header(unsigned char* buffer, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order,
                          std::string& typeName);
NpyArray npy_load(std::string fname);

template <typename T>
std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs)
{
  // write in little endian
  for (size_t byte = 0; byte < sizeof(T); byte++) {
    char val = *((char*)&rhs + byte);
    lhs.push_back(val);
  }
  return lhs;
}

template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs);
template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs);

template <typename T>
int npy_save(std::string fname, const T* data, const std::vector<size_t> shape, std::string mode = "w")
{
  std::ofstream ofs(fname, std::ios::out);
  if (!ofs.is_open()) {
    return -1;
  }
  ofs.close();
  FILE*               fp = NULL;
  std::vector<size_t> true_data_shape; // if appending, the shape of existing + new data

  if (mode == "a")
    fp = fopen(fname.c_str(), "r+b");

  if (fp) {
    // file exists. we need to append to it. read the header, modify the array size
    size_t      word_size;
    bool        fortran_order;
    std::string typeName;
    parse_npy_header(fp, word_size, true_data_shape, fortran_order, typeName);
    assert(!fortran_order);

    if (word_size != sizeof(T)) {
      std::cout << "libnpy error: " << fname << " has word size " << word_size << " but npy_save appending data sized "
                << sizeof(T) << "\n";
      assert(word_size == sizeof(T));
    }
    if (true_data_shape.size() != shape.size()) {
      std::cout << "libnpy error: npy_save attempting to append misdimensioned data to " << fname << "\n";
      assert(true_data_shape.size() != shape.size());
    }

    for (size_t i = 1; i < shape.size(); i++) {
      if (shape[i] != true_data_shape[i]) {
        std::cout << "libnpy error: npy_save attempting to append misshaped data to " << fname << "\n";
        assert(shape[i] == true_data_shape[i]);
      }
    }
    true_data_shape[0] += shape[0];
  } else {
    fp              = fopen(fname.c_str(), "wb");
    true_data_shape = shape;
  }

  std::vector<char> header = create_npy_header<T>(true_data_shape);
  size_t            nels   = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

  fseek(fp, 0, SEEK_SET);
  fwrite(&header[0], sizeof(char), header.size(), fp);
  fseek(fp, 0, SEEK_END);
  fwrite(data, sizeof(T), nels, fp);
  fclose(fp);
  return 0;
}

template <typename T>
void npy_save(std::string fname, const std::vector<T> data, std::string mode = "w")
{
  std::vector<size_t> shape;
  shape.push_back(data.size());
  npy_save(fname, &data[0], shape, mode);
}

template <typename T>
std::vector<char> create_npy_header(const std::vector<size_t>& shape)
{
  const char* tpye_name = typeid(T).name();
  std::vector<char> dict;
  dict += "{'descr': '";
  dict += BigEndianTest(sizeof(T));
  if (std::string(tpye_name) == "N4rknn7float16E") {
    dict += "f";
  } else {
    dict += map_type(typeid(T));
  }
  dict += std::to_string(sizeof(T));
  dict += "', 'fortran_order': False, 'shape': (";
  dict += std::to_string(shape[0]);
  for (size_t i = 1; i < shape.size(); i++) {
    dict += ", ";
    dict += std::to_string(shape[i]);
  }
  if (shape.size() == 1)
    dict += ",";
  dict += "), }";
  // pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
  int remainder = 16 - (10 + dict.size()) % 16;
  dict.insert(dict.end(), remainder, ' ');
  dict.back() = '\n';

  std::vector<char> header;
  header += (char)0x93;
  header += "NUMPY";
  header += (char)0x01; // major version of numpy format
  header += (char)0x00; // minor version of numpy format
  header += (uint16_t)dict.size();
  header.insert(header.end(), dict.begin(), dict.end());

  return header;
}

} // namespace cnpy

#endif
