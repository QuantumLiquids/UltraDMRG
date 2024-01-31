// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-14
*
* Description: GraceQ/mps2 project. Lanczos performance test on single processor
*/

/**
 * @note this file only compile/link in cmakes, won't add to tests; users should run it by self
 * @note only support U1U1 quantum number and double
 */


#include "gqmps2/algorithm/lanczos_params.h"
#include "../tests/testing_utils.h"
#include "gqten/gqten.h"
#include "gqten/utility/timer.h"
#include "gperftools/profiler.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <thread>


#ifdef Release
  #define NDEBUG
#endif

#include <assert.h>

#include "mkl.h"


using namespace gqmps2;
using namespace gqten;

using std::vector;
using std::cout;
using std::endl;
using std::string;

using U1U1QN = QN<U1QNVal, U1QNVal>;
using IndexT = Index<U1U1QN>;
using QNSctT = QNSector<U1U1QN>;
using QNSctVecT = QNSectorVec<U1U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1U1QN>;


const std::vector<size_t> default_thread_nums =  {5,10,20,40,60,80};

//forward declaration
int Parser(int argc, char *argv[],
            std::string& path,
            vector<size_t>& thread_nums);


/**
 * 
 * 
 * @param argv    --path=[the working directory], e.g. --path=./tensor_package; if no this arguement, --path=./ by default.
 * @param argv    --thread_nums=[a list of number of threads], e.g. --thread_nums=12,24,48,96, note no space between this arguement.
 *                 by default, --thread_nums=5,10,20,40,60,80.
 * 
 */
int main(int argc, char *argv[]){
  std::string working_path;
  std::vector<size_t> thread_nums;
  Parser(argc, argv, working_path, thread_nums);
  cout << "thread number list: [ ";
  for(size_t i=0;i < thread_nums.size(); i++){
    cout << thread_nums[i];
    if(i<thread_nums.size()-1){
      cout << ", ";
    }else{
      cout << "]\n";
    }
  }
  
  //Loading file
  DGQTensor lenv, renv, mpo1, mpo2, mps1, mps2;
  vector<DGQTensor *> load_ten_list = {&lenv, &renv, &mpo1, &mpo2, &mps1, &mps2};
  vector<std::string > file_name_list = {"lenv.gqten", "renv.gqten", "mpo_ten_l.gqten",
                              "mpo_ten_r.gqten", "mps_ten_l.gqten", "mps_ten_r.gqten"};
  for(size_t i =0;i<load_ten_list.size();i++){
    std::string file = working_path + file_name_list[i];
    if( access( file.c_str(), 4) != 0){
      std::cout << "The progress doesn't access to read the file " << file << "!" << std::endl;
      exit(1);
    }
    std::ifstream ifs(file, std::ios::binary);
    if(!ifs.good()){
      std::cout << "The progress can not read the file " << file << " correctly!" << std::endl;
      exit(1);
    }
    ifs >> *load_ten_list[i];
  }
  std::cout << "The progress has loaded the tensors." <<std::endl;
  cout << "Concise Info of tensors: \n";
  cout << "lenv.gqten:"; lenv.ConciseShow();
  cout << "renv.gqten:"; renv.ConciseShow();
  cout << "mpo_ten_l.gqten:"; mpo1.Show();
  cout << "mpo_ten_r.gqten:"; mpo2.Show();
  cout << "mps_ten_l.gqten:"; mps1.ConciseShow();
  cout << "mps_ten_r.gqten:"; mps2.ConciseShow();

  if(lenv.GetIndexes()[0] != mps1.GetIndexes()[0]) { //old version data, for compatible
    assert(lenv.GetIndexes()[0] == InverseIndex(mps1.GetIndexes()[0]));
    lenv.Transpose({2,1,0});
    renv.Transpose({2,1,0});
  }

  //Get the initial state
  vector<DGQTensor*> eff_ham = {&lenv, &mpo1, &mpo2, &renv};
  DGQTensor* state = new DGQTensor();
  hp_numeric::SetTensorManipulationTotalThreads(14);
  hp_numeric::SetTensorTransposeNumThreads(14);
  Contract(&mps1, &mps2, {{2},{0}}, state);

  cout << "initial state:"; state->ConciseShow();

  const size_t max_threads = std::thread::hardware_concurrency();
  for(size_t i=0;i<thread_nums.size();i++){
    cout << "[test case " << i << " ]\n";
    if(thread_nums[i] > max_threads  ){
      std::cout << "warning: maximal thread are " << max_threads <<", but require thread " << thread_nums[i] <<"." << std::endl;
      std::cout << "test case passed." <<std::endl;
      continue;
    }
    std::string profiler_report_file = "single_process_lanczos_thread" + std::to_string(thread_nums[i]) + ".o";
    hp_numeric::SetTensorManipulationTotalThreads(thread_nums[i]);
    hp_numeric::SetTensorTransposeNumThreads(thread_nums[i]);
    ProfilerStart( profiler_report_file.c_str() );

    #pragma omp parallel num_threads(hp_numeric::tensor_manipulation_num_threads) default(none)
    {
      if(omp_get_thread_num() > 0) {
        ProfilerDisable(); //Only measure the main thread
      }
    }
    Timer single_process_mat_vec_timer("single processor lanczos matrix multiply vector (thread " + std::to_string(thread_nums[i]) + ")");
    DGQTensor* single_process_res = eff_ham_mul_two_site_state(eff_ham, state);
    single_process_mat_vec_timer.PrintElapsed();
    delete single_process_res;
    //Run again
    single_process_mat_vec_timer.ClearAndRestart();
    single_process_res = eff_ham_mul_two_site_state(eff_ham, state);
    single_process_mat_vec_timer.PrintElapsed();
    delete single_process_res;
    ProfilerStop();
  }

  std::ofstream ofs("state.gqten", std::ios::binary );
  ofs << *state;
  delete state;
  return 0;
}

int Parser(int argc, char *argv[],
            std::string& path,
            vector<size_t>& thread_nums){
int nOptionIndex = 1;
string thread_nums_string;
 
string arguement1 = "--path=";
string arguement2 = "--thread_nums=";
while (nOptionIndex < argc){
  if (strncmp(argv[nOptionIndex], arguement1.c_str() , arguement1.size()) == 0){
    path = &argv[nOptionIndex][arguement1.size()];  // path
  }else if (strncmp(argv[nOptionIndex], arguement2.c_str(), arguement2.size()) == 0){
    thread_nums_string = &argv[nOptionIndex][arguement2.size()];// thread number list
  }
  else{
    cout << "Options '" << argv[nOptionIndex] << "' not valid. Run '" << argv[0] << "' for details." << endl;
  //   return -1;
  }
  nOptionIndex++;
}

//split thread num list
const char* split = ",";
char *p;
const size_t MAX_CHAR_LENTH = 1000;
char thread_nums_char[MAX_CHAR_LENTH];
for(size_t i =0;i< MAX_CHAR_LENTH;i++){
  thread_nums_char[i] = 0;
}

strcpy(thread_nums_char, thread_nums_string.c_str() );

p = strtok(thread_nums_char, split);
while(p != nullptr){
  thread_nums.push_back( atoi(p) );
  p = strtok(nullptr, split);
}


if(thread_nums.size() == 0){
  thread_nums = default_thread_nums; //default thread numbers
}

if(path.size() == 0){
  path = "./";
}
 
return 0;

}