#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <variant>
#include <tuple>
#include <set>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <random>
#include <queue>
#include <chrono>

using std::vector;
using std::tuple;
using std::set;
using std::map;
using std::optional;
using std::string;

#define DOUT(x) \
  std::cout << x << std::endl;
#define DLINEOUT(x) \
  std::cout << "Line " << __LINE__ << " | " << x << std::endl;
#define DLINE \
  DLINEOUT(' ')
#define DLINEFILEOUT(x) \
  std::cout << __FILE__ << " @ " << __LINE__ << " | " << x << std::endl;
#define DLINEFILE \
  DLINEFILEOUT(' ')

#define vector_from_each_member(items, member_type, member_name) [](auto const& xs) { \
    std::vector<member_type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return x.member_name; }); \
    return ret; \
  }(items)

#define vector_from_each_method(items, type, method) [](auto const& xs) { \
    std::vector<type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return x.method(); }); \
    return ret; \
  }(items)

#define vector_from_each_tuple(items, which_type, which) [](auto const& xs) { \
    std::vector<which_type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return std::get<which>(x); }); \
    return ret; \
  }(items)


template <typename T>
bool vector_equal(vector<T> const& xs, vector<T> const& ys) {
  if(xs.size() != ys.size()) {
    return false;
  }
  for(int i = 0; i != xs.size(); ++i) {
    if(xs[i] != ys[i]) {
      return false;
    }
  }

  return true;
}

template <typename T>
T product(vector<T> const& xs)
{
  T ret = 1;
  for(T const& x: xs) {
    ret *= x;
  }
  return ret;
}

template <typename T>
void print_vec(vector<T> const& xs)
{
  print_vec(std::cout, xs);
}

template <typename T>
void print_vec(std::ostream& out, vector<T> const& xs)
{
  out << "{";
  if(xs.size() >= 1) {
    out << xs[0];
  }
  if(xs.size() > 1) {
    for(int i = 1; i != xs.size(); ++i) {
      out << "," << xs[i];
    }
  }
  out << "}";
}

template <typename T>
[[nodiscard]] vector<T> vector_concatenate(vector<T> vs, vector<T> const& add_these) {
  vs.reserve(vs.size() + add_these.size());
  for(auto const& x: add_these) {
    vs.push_back(x);
  }
  return vs;
}
template <typename T>
void vector_concatenate_into(vector<T>& vs, vector<T> const& add_these) {
  vs.reserve(vs.size() + add_these.size());
  for(auto const& x: add_these) {
    vs.push_back(x);
  }
}

template <typename T>
std::ostream& operator<<(std::ostream& out, vector<T> const& ts) {
  print_vec(out, ts);
  return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, set<T> const& ts) {
  // TODO: implement print_set
  print_vec(out, vector<T>(ts.begin(), ts.end()));
  return out;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& out, tuple<T, U> const& x12) {
  auto const& [x1,x2] = x12;
  out << "tup[" << x1 << "|" << x2 << "]";
  return out;
}

std::mt19937& random_gen() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  return gen;
}

void set_seed(int seed) {
  random_gen() = std::mt19937(seed);
}

template <typename T>
using priority_queue_least = std::priority_queue<T, vector<T>, std::greater<T>>;
// For priority_queue_least, the top most element is the smallest,
// which is the opposite behaviour of priority_queue which puts the
// largest element at the top.

#define clock_now std::chrono::high_resolution_clock::now

using timestamp_t = decltype(clock_now());

struct raii_print_time_elapsed_t {
  raii_print_time_elapsed_t(string msg):
    msg(msg), start(clock_now()), out(std::cout)
  {}

  raii_print_time_elapsed_t():
    msg(), start(clock_now()), out(std::cout)
  {}

  ~raii_print_time_elapsed_t() {
    auto end = clock_now();
    using namespace std::chrono;
    auto duration = (double) duration_cast<microseconds>(end - start).count()
                  / (double) duration_cast<microseconds>(1s         ).count();

    if(msg.size() > 0) {
      out << msg << " | ";
    }
    out << "Total Time (seconds): " << duration << std::endl;
  }

  string const msg;
  timestamp_t const start;
  std::ostream& out;
};

using gremlin_t = raii_print_time_elapsed_t;

// Stolen from http://myeyesareblind.com/2017/02/06/Combine-hash-values/
// where this is the boost implementation
void hash_combine_impl(std::size_t& seed, std::size_t value)
{
    seed ^= value + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

template <typename T>
vector<T> _reverse_variadic_to_vec(T i) {
  vector<T> x(1, i);
  return x;
}
template <typename T, typename... Args>
vector<T> _reverse_variadic_to_vec(T i, Args... is) {
  vector<T> x = _reverse_variadic_to_vec(is...);
  x.push_back(i);
  return x;
}

template <typename T, typename... Args>
vector<T> variadic_to_vec(Args... is) {
  vector<T> x = _reverse_variadic_to_vec(is...);
  std::reverse(x.begin(), x.end());
  return x;
}

template <typename T>
optional<string> check_concat_shapes(
  int dim,
  vector<vector<T>> const& shapes)
{
  if(shapes.size() == 0) {
    return "cannot be empty list of shapes";
  }

  // they should all have the same rank
  int rank = shapes[0].size();
  for(int i = 1; i != shapes.size(); ++i) {
    if(shapes[i].size() != rank) {
      return "invalid input size";
    }
  }

  if(dim < 0 || dim >= rank) {
    return "invalid dim";
  }

  // every dim should be the same, except dim
  vector<T> dim_parts;
  for(int r = 0; r != rank; ++r) {
    if(r != dim) {
      T d = shapes[0][r];
      for(int i = 1; i != shapes.size(); ++i) {
        if(shapes[i][r] != d) {
          return "non-concat dimensions do not line up";
        }
      }
    }
  }

  return std::nullopt;
}

template <typename T, typename U>
vector<T> vector_mapfst(vector<tuple<T, U>> const& xys) {
  return vector_from_each_tuple(xys, T, 0);
}

template <typename T, typename U>
vector<U> vector_mapsnd(vector<tuple<T, U>> const& xys) {
  return vector_from_each_tuple(xys, T, 1);
}

template <typename T>
T parse_with_ss(string const& s)
{
  T out;
  std::istringstream ss(s);
  ss >> out;
  return out;
}

template <typename T>
string write_with_ss(T const& val)
{
  std::ostringstream ss;
  ss << val;
  return ss.str();
}

// Find the last true element
// Assumption: evaluate returns all trues then all falses.
// If there are no trues: return end
// If there are all trues: return end-1
template <typename Iter, typename F>
Iter binary_search_find(Iter beg, Iter end, F evaluate)
{
  if(beg == end) {
    return end;
  }
  if(!evaluate(*beg)) {
    return end;
  }

  decltype(std::distance(beg,end)) df;
  while((df = std::distance(beg, end)) > 2) {
    Iter mid = beg + (df / 2);
    if(evaluate(*mid)) {
      beg = mid;
    } else {
      end = mid;
    }
  }

  if(df == 1) {
    return beg;
  }

  if(evaluate(*(end - 1))) {
    return end-1;
  } else {
    return beg;
  }
}


