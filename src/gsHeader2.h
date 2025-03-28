/** @file gsHeader2.h

    @brief Example Class

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): P. Weinmueller
*/

#pragma once
#include <gismo.h>


namespace gismo
{
  /**
    \brief I am an example how to use the class in submodules
*/

class submoduleClass2
{
  
public:
  // Empty constructor
  ~submoduleClass2() {}

  void printString(std::string const & str) const
  {
    gsInfo << str << "\n";
  }

}; // class submoduleClass2

} // namespace gismo
