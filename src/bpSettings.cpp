/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "bp.hpp"

//settings for bp solver
bpSettings_t::bpSettings_t(const int argc, char** argv, comm_t _comm):
  settings_t(_comm) {

  platformAddSettings(*this);
  meshAddSettings(*this);

  newSetting("-P", "--problem",
             "BENCHMARK PROBLEM",
             "1",
             "Benchmark problem to run",
             {"1","2","3","4","5","6"});

  newToggle("-k", "--kernel",
            "KERNEL TEST",
            "FALSE",
            "Benchmark only operator kernel (i.e. run BK)");

  newToggle("-t", "--tuning",
            "KERNEL TUNING",
            "FALSE",
            "Run tuning sweep on operator kernel");

  newSetting("-o", "--output",
             "OUTPUT FILE NAME",
             "");

  newToggle("-v", "--verbose",
            "VERBOSE",
            "FALSE",
            "Enable verbose output");

  parseSettings(argc, argv);
}

void bpSettings_t::report() {

  if (comm.rank()==0) {
    std::cout << "Settings:\n\n";
    platformReportSettings(*this);
    meshReportSettings(*this);

    if (getSetting("OUTPUT FILE NAME").size()>0)
      reportSetting("OUTPUT FILE NAME");
  }
}
