/** @file gsHeader1.h

    @brief Example Class

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): P. Weinmueller

SEE ALSO: https://github.com/gismo/gismo/wiki/Contributing#code-style-and-formatting
*/

#pragma once
#include <gismo.h>


namespace gismo
{
  /**
    \brief I am an example how to use the class in submodules
*/

class submoduleClass
{
  
public:
  // Empty constructor
  ~submoduleClass() {}

  void printString(std::string const & str) const
  {
    gsInfo << str << "\n";
  }

}; // class submoduleClass

} // namespace gismo
