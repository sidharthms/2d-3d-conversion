--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'
local debugger = require 'fb.debugger'

local M = {}

function M.setupCriterion(opt)
   local criterion = nn.AbsCriterion():cuda()
   return criterion
end

function M.setup(opt, checkpoint)
   print('=> Creating model from file: models/' .. opt.net_type .. '.lua')
   local model, featureModel = require('models/' .. opt.net_type)(opt)

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.share_grad_input then
      if opt.optnet then
         error('not supported with optnet')
      end
      print('Sharing grad inputs')
      M.shareGradInput(featureModel)
      M.shareGradInput(model)
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   -- if opt.nGPU > 1 then
   --    debugger.enter()
   --    local gpus = torch.range(1, opt.nGPU):totable()
   --    local fastest, benchmark = cudnn.fastest, cudnn.benchmark
   --    cudnn.fastest, cudnn.benchmark = fastest, benchmark
   --
   --    local dpt = nn.DataParallelTable(1, true, true)
   --       :add(model, gpus)
   --    dpt.gradInput = nil
   --
   --    model = dpt:cuda()
   -- end

   local criterion = M.setupCriterion(opt)
   return model, featureModel, criterion
end

function M.shareGradInput(model)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch.CudaStorage(1)
      end
      m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
   end
end

return M
