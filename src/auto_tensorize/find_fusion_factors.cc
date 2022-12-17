#include <auto_compute/graph.h>
#include <auto_tensorize/analysis.h>
#include <auto_tensorize/dse/searchDriver.h>
#include <auto_tensorize/hyper_fusion.h>
#include <auto_tensorize/iter_graph.h>
#include <auto_tensorize/state.h>
#include <tvm/te/schedule_pass.h>

using namespace std;

#define ceil(a, b) ((a+b-1)/b*b)

namespace ditto {
namespace auto_tensorize{
    void setTemplatesAndSearchByRule(
        const IterGraph& ig, const hardware::HardwareParam& hw_param, int bytePerEle,
        std::vector<FusionInfo>* fusion_infos_ptr, bool verbose
    ){
        // fusion_info.bounds, fusion_info.fusionLevel
        // fusion_info.cache_occupancy
        // fusion_info.secondOpOuterIndices
        // fusion_info.secondOpOuterTilingFactors
        // fusion_info.parallelFactor
        Array<IntImm> firstOpPermute, firstOpTiling;
        size_t idx = 0;
        for (auto iv : ig->_firstOpIters)
        {
            firstOpPermute.push_back(IntImm(DataType::Int(32), idx++));
            firstOpTiling.push_back(IntImm(DataType::Int(32), 1));
        }
        ig->setFirstOpPermute(firstOpPermute);
        ig->setFirstOpTiling(firstOpTiling);

        vector<int> op1S1, op1S2, op1R, op1B, op2S1, op2S2, op2B, op2R;
        for (size_t i = 0; i < ig->_firstOpIters.size(); i++){
            auto iv = ig->_firstOpIters[i];
            if (iv->iv_type == IV_Type::FIRSTSPATIAL) op1S1.push_back(i);
            else if (iv->iv_type == IV_Type::SECONDSPATIAL) op1S2.push_back(i);
            else if (iv->iv_type == IV_Type::BATCH) op1B.push_back(i);
            else if (iv->iv_type == IV_Type::REDUCE) op1R.push_back(i);
            else CHECK(false) << "invalid iv type";
        }
        for (size_t i = 0; i < ig->_secondOpIters.size(); i++){
            auto iv = ig->_secondOpIters[i];
            if (iv->iv_type == IV_Type::FIRSTSPATIAL) op2S1.push_back(i);
            else if (iv->iv_type == IV_Type::SECONDSPATIAL) op2S2.push_back(i);
            else if (iv->iv_type == IV_Type::BATCH) op2B.push_back(i);
            else if (iv->iv_type == IV_Type::REDUCE) op2R.push_back(i);
            else CHECK(false) << "invalid iv type";
        }


        int m_tensorize = 1, k_tensorize = 1, l_tensorize = 1, n_tensorize = 1;
        int M, N, K, L, B;
        M = N = K = L = B = 1;
        //! TODO: Is there case that the ivs have non-dicar style?
        for (auto idx: op1S1) {
            m_tensorize *= ig->_firstOpIters[idx]->tensorize_ext;
            M *= ig->_firstOpIters[idx]->ext;
        }
        for (auto idx: op1R) {
            k_tensorize *= ig->_firstOpIters[idx]->tensorize_ext;
            K *= ig->_firstOpIters[idx]->ext;
        }
        for (auto idx: op1S2) {
            l_tensorize *= ig->_firstOpIters[idx]->tensorize_ext;
            L *= ig->_firstOpIters[idx]->ext;
        }
        for (auto idx: op2S2) {
            n_tensorize *= ig->_secondOpIters[idx]->tensorize_ext;
            N *= ig->_secondOpIters[idx]->ext;
        }
        for (auto idx: op1B) B *= ig->_firstOpIters[idx]->ext;

        Array<IntImm> secondOpPermute_;
        for (auto & ivs: {op2B, op2S1, op2R, op2S2}) 
            for (auto idx: ivs) 
                secondOpPermute_.push_back(IntImm(DataType::Int(32), idx));
        ig->setSecondOpPermute(secondOpPermute_);
        if (verbose){
            cout << "set second permute: ";
            for (auto idx: secondOpPermute_){
                cout << "[" << ig->_secondOpIters[idx->value] << "|" << idx->value << "],";
            }
            cout << endl;
            cout << "second op iters: " << ig->secondOpIters << endl;
        }
        ig->setAttach(op2B.size() + op2S1.size() + op2R.size());

        double alpha = max(n_tensorize, k_tensorize);
        ig->setConfig(hw_param, bytePerEle);
        CHECK(fusion_infos_ptr != nullptr);
        int id = 0;
        for (auto fusion_level: ig->fusionLevels){
            ig->setFusionLevel(fusion_level);
            double MC = hw_param->cacheSizes[fusion_level] / bytePerEle;
            int opt = min((-alpha + sqrt(alpha * alpha + MC)), sqrt((M * L + 0.0) / hw_param->num_groups));
            
            int Tm = min(M, max(opt / m_tensorize, 1));
            int Tl = min(L, max(opt / l_tensorize, 1));
            int parallel = min(B * ((M + Tm - 1) / Tm) * ((L + Tl - 1) / Tl), hw_param->num_groups);
            double cost = bytePerEle * B * (M * K * m_tensorize * k_tensorize * ceil(L, Tl) 
                        + K * L * k_tensorize * l_tensorize * ceil(M, Tm)
                        + N * L * n_tensorize * l_tensorize * ceil(M, Tm)
                        + M * N * m_tensorize * n_tensorize * ceil(L, Tl)) / hw_param->cacheBandwidth[fusion_level] / parallel;
            double occupancy = (Tm * m_tensorize * alpha + 
                                Tl * l_tensorize * alpha +
                                Tm * Tl * m_tensorize * l_tensorize)/hw_param->cacheSizes[fusion_level] / bytePerEle;
            if (verbose){
                fprintf(stdout, "[search template]: fusionLevel %d, cost %f, MC %f, alpha %f\n", 
                    fusion_level, cost, MC, alpha);
                fprintf(stdout, "\tBMNKL: %d,%d,%d,%d,%d;t_MNKL:%d,%d,%d,%d\n", B,M,N,K,L,m_tensorize,n_tensorize,k_tensorize,l_tensorize);
                fprintf(stdout, "\topt:%d,Tm:%d,Tl:%d,parallel:%d\n", opt, Tm, Tl, parallel);
            }
            FusionInfo info;
            info.cost = cost;
            info.parallelism = parallel;
            info.fusionLevel = fusion_level;
            info.cacheOccupancy = occupancy;
            // share the tilesize of Tm
            int tile_size = Tm;
            vector<int> secondOpTilingFactors(ig->_secondOpIters.size(), 1);
            for (auto idx: op2S1) {
                auto iv = ig->_secondOpIters[idx];
                info.secondOpOuterIndices.push_back(iv->index);
                int this_tile_size = min(tile_size, iv->ext);
                CHECK(this_tile_size > 0);
                info.secondOpOuterTilingFactors.push_back(this_tile_size);
                tile_size = (tile_size + this_tile_size - 1) / this_tile_size;
                secondOpTilingFactors[idx] = this_tile_size;
            }
            tile_size = Tl;
            for (auto idx: op2R) {
                auto iv = ig->_secondOpIters[idx];
                info.secondOpOuterIndices.push_back(iv->index);
                int this_tile_size = min(tile_size, iv->ext);
                CHECK(this_tile_size > 0);
                info.secondOpOuterTilingFactors.push_back(this_tile_size);
                tile_size = (tile_size + this_tile_size - 1) / this_tile_size;
                secondOpTilingFactors[idx] = this_tile_size;
            }
            Array<IntImm> _secondOpTilingFactors;
            for (auto factor: secondOpTilingFactors){
                _secondOpTilingFactors.push_back(IntImm(DataType::Int(32),factor));
            }
            if (verbose) {
                cout << "set second op tiling";
                for (size_t i = 0; i < ig->_secondOpIters.size(); i++){
                    cout << "[" << ig->_secondOpIters[i] << "|" << _secondOpTilingFactors[i] << "], ";
                }
                cout << endl;
            }
            ig->setSecondOpTiling(_secondOpTilingFactors);
            ig->applyAll();
            info.lower_bounds_for_upper_cache_level = ig->inferBound();
            ig->scheduleParallel();
            if (verbose) {
                ig->visualize();
            }
            info.boundsAfterParallel = ig->_boundsAfterParallel;
            info.parallelFactor = ig->_parallelSchedule;
            info.upper_bounds_for_lower_cache_level = ig->inferBound();
            info.valid = true;
            // set op1R and op2S2 to 1
            for (auto idx: op1R){
                auto iv = ig->_firstOpIters[idx];
                info.upper_bounds_for_lower_cache_level.Set(iv->originVar, IntImm(DataType::Int(32), 1));
                info.op1InnerIvs.push_back({iv->index, iv->ext});
            }
            for (auto idx: op2S2){
                auto iv = ig->_secondOpIters[idx];
                info.upper_bounds_for_lower_cache_level.Set(iv->originVar, IntImm(DataType::Int(32), 1));
                info.op2InnerIvs.push_back({iv->index, iv->ext});
            }
            // op1R and op2S2
            (*fusion_infos_ptr)[id++] = move(info);
        }
        fusion_infos_ptr->resize(id);
        sort (fusion_infos_ptr->begin(), fusion_infos_ptr->end(), [](const FusionInfo& info1, const FusionInfo& info2){
            return info1.cost < info2.cost;
        });
    }
} // namespace auto_tensorize
} // namespace ditto