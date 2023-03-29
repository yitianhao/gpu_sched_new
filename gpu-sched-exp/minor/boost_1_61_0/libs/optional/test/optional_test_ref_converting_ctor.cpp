// Copyright (C) 2014 Andrzej Krzemienski.
//
// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/lib/optional for documentation.
//
// You are welcome to contact the author at: akrzemi1@gmail.com


#include "boost/optional/optional.hpp"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#include "boost/core/addressof.hpp"
#include "boost/core/enable_if.hpp"
#include "boost/core/lightweight_test.hpp"
#include "testable_classes.hpp"

using boost::optional;
using boost::none;

template <typename T>
void test_converting_ctor()
{
  typename concrete_type_of<T>::type v1(1), v2(2);
  
  {
    optional<T&> o1 = v1, o1_ = v1, o2 = v2;
    
    BOOST_TEST(o1);
    BOOST_TEST(boost::addressof(*o1) == boost::addressof(v1));
    BOOST_TEST(o1_);
    BOOST_TEST(boost::addressof(*o1_) == boost::addressof(v1));
    BOOST_TEST(boost::addressof(*o1_) == boost::addressof(*o1));
    
    BOOST_TEST(o2);
    BOOST_TEST(boost::addressof(*o2) == boost::addressof(v2));
    BOOST_TEST(boost::addressof(*o2) != boost::addressof(*o1));
  }
  {
    const optional<T&> o1 = v1, o1_ = v1, o2 = v2;
    
    BOOST_TEST(o1);
    BOOST_TEST(boost::addressof(*o1) == boost::addressof(v1));
    BOOST_TEST(o1_);
    BOOST_TEST(boost::addressof(*o1_) == boost::addressof(v1));
    BOOST_TEST(boost::addressof(*o1_) == boost::addressof(*o1));
    
    BOOST_TEST(o2);
    BOOST_TEST(boost::addressof(*o2) == boost::addressof(v2));
    BOOST_TEST(boost::addressof(*o2) != boost::addressof(*o1));
  }
}

template <typename T>
void test_converting_ctor_for_noconst_const()
{
  typename concrete_type_of<T>::type v1(1), v2(2);
  
  {
    optional<const T&> o1 = v1, o1_ = v1, o2 = v2;
    
    BOOST_TEST(o1);
    BOOST_TEST(boost::addressof(*o1) == boost::addressof(v1));
    BOOST_TEST(o1_);
    BOOST_TEST(boost::addressof(*o1_) == boost::addressof(v1));
    BOOST_TEST(boost::addressof(*o1_) == boost::addressof(*o1));
    
    BOOST_TEST(o2);
    BOOST_TEST(boost::addressof(*o2) == boost::addressof(v2));
    BOOST_TEST(boost::addressof(*o2) != boost::addressof(*o1));
  }
  {
    const optional<const T&> o1 = v1, o1_ = v1, o2 = v2;
    
    BOOST_TEST(o1);
    BOOST_TEST(boost::addressof(*o1) == boost::addressof(v1));
    BOOST_TEST(o1_);
    BOOST_TEST(boost::addressof(*o1_) == boost::addressof(v1));
    BOOST_TEST(boost::addressof(*o1_) == boost::addressof(*o1));
    
    BOOST_TEST(o2);
    BOOST_TEST(boost::addressof(*o2) == boost::addressof(v2));
    BOOST_TEST(boost::addressof(*o2) != boost::addressof(*o1));
  }
}

template <typename T>
void test_all_const_cases()
{
  test_converting_ctor<T>();
  test_converting_ctor<const T>();
  test_converting_ctor_for_noconst_const<T>();
}

int main()
{
  test_all_const_cases<int>();
  test_all_const_cases<ScopeGuard>();
  test_all_const_cases<Abstract>();
  
  return boost::report_errors();
}
