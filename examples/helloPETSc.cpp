/** @file helloSubmodule.cpp

    @brief First example of submodule

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): P. Weinmueller
*/

//! [Include namespace]
#include <gismo.h>

#include <gsModule/gsTemplates/gsClass.h>
#include <gsModule/gsHeader2.h>

using namespace gismo;
//! [Include namespace]

int main(int argc, char *argv[])
{

  gsInfo << "Hello submodule!\n";

  // Example how to use the class in submodules
  submoduleClass submClass;
  std::string str = "Hello, I am called from the submodule class, stored in gsSubmodule/gsFolder1";
  submClass.printString(str);
  
  submoduleClass2 submClass2;
  std::string str2 = "Hello, I am called from the submodule class, stored in gsSubmodule";
  submClass2.printString(str2);
  
  gsMultiPatch<real_t> mp;
  gsReadFile<>("turtle.xml",mp);
  gsInfo<<"Plotting the geometry, given in gsSubmodule/filedata in Paraview as geom.pvd\n";
  gsWriteParaview( mp, "geom",1000,true);
  
  return EXIT_SUCCESS;
}
