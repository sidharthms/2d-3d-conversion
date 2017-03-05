--
-- Created by IntelliJ IDEA.
-- User: sidharth
-- Date: 3/11/16
-- Time: 11:04 PM
-- To change this template use File | Settings | File Templates.
--
--
-- Copyright (c) 2016, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.
--

local stringx = require('pl.stringx')
local path = require 'pl.path'

local M = { }

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Entity Matching Training script')
cmd:text()
cmd:text('Options:')

------------ General options --------------------
cmd:option('-saved_model', '', 'pretrained product model file')
cmd:option('-data_dir','','data directory.')
cmd:option('-no_compress',false,'')
cmd:option('-proddata_dir','proddata.t7','data directory.')
cmd:option('-gpuid', 0,'which gpu to use. -1 = use CPU')
cmd:option('-n_gpu', 1, 'Number of GPUs to use by default')
cmd:option('-savefile', '', 'path to save model')
cmd:option('-nosave', false, 'dont save model')
cmd:option('-save_every_epoch', false, 'save every epoch to disk')
cmd:option('-manual_seed', 2, 'Manually set RNG seed')
cmd:option('-batch_limit', -1, 'Manually set RNG seed')
cmd:option('-no_print_stats', false, 'Don\'t print stats while training')
cmd:option('-unmonitored_mode', false, 'for spearmint')
cmd:option('-cudnn', 'fastest', 'Options: fastest | default | deterministic')
cmd:option('-samples_dir', '', 'Dir for samples')
cmd:option('-sample_save_rate', 50, '')

------------- Data options ------------------------
cmd:option('-n_threads',        4, 'number of data loading threads')

------------- Checkpointing options ---------------
cmd:option('-resume', 'none', 'Resume from the latest checkpoint in this directory')

------------- Training options --------------------
cmd:option('-batch_size', 16, 'number of products to train on in parallel')
cmd:option('-lr_decay_period', 2, 'number of epochs between each reduction in lr')
cmd:option('-lr_decay_factor', 0.5, 'lr decay factor')
cmd:option('-epochs', 50, 'number of full passes through the training data')
cmd:option('-no_eval', false, 'Manually set RNG seed')
cmd:option('-data_split_size', 32, 'clip gradients to limit grad norm')
cmd:option('-parallel_preprocessing', false, 'use parallel thread to load data')
cmd:option('-train_mats', 5, 'use parallel thread to load data')
cmd:option('-num_batches',  '', 'Share gradInput tensors to reduce memory usage')
cmd:option('-batch_size_factor',  1, 'Share gradInput tensors to reduce memory usage')

---------- Optimization options ----------------------
cmd:option('-optim', 'adam', 'optimization method')
cmd:option('-lr', 0.01, 'initial learning rate')
cmd:option('-beta1', 0.9, 'adam momentum')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-weight_decay', 0, 'weight decay')
cmd:option('-batch_randperm', false, 'use random perm')
cmd:option('-clip_grads', false, 'clip gradients to limit grad norm')
cmd:option('-grad_max_norm', 5, 'max norm of gradients')
cmd:option('-grad_max', 100, 'max gradient')
cmd:option('-d_lr_factor',  1, '')
cmd:option('-d_update_freq',  1, '')
cmd:option('-gan_l1_factor',  100, '')

---------- Model options ----------------------------------
cmd:option('-net_type', 'arnn', 'Model type')
cmd:option('-disp_hidden_size', 64, 'size of disp map hidden size')
cmd:option('-regress_disparity', false, 'regress against disparity map instead of right image')
cmd:option('-disparity_range', 33, 'range of disparity')
cmd:option('-resnet_model','','path to pretrained resnet model')
cmd:option('-deep3d_base','','path to pretrained deep3d model')
cmd:option('-xsize', 384, 'xsize')
cmd:option('-ysize', 160, 'ysize')
cmd:option('-optnet', false, 'Use optnet to reduce memory usage')
cmd:option('-share_grad_input',  false, 'Share gradInput tensors to reduce memory usage')
cmd:option('-batchnorm_after_upsample',  false, '')
cmd:option('-relu_after_upsample',  false, '')
cmd:option('-kaimint_init',  false, '')
cmd:option('-disp_upsample_range',  1, '')
cmd:option('-gan_model',  '', '')
cmd:option('-no_cgan',  false, '')
cmd:option('-gan_d_hidden_size',  64, '')
cmd:option('-gan_d_extra_layer',  false, '')
cmd:option('-magicpony_upsample',  false, '')
cmd:option('-magicpony_feature_upsample_kernel',  -1, '')
cmd:option('-magicpony_highres_upsample_kernel',  3, '')
cmd:option('-highres_batchnorm',  false, '')
cmd:text()

function M.parse(arg)
   local opt = cmd:parse(arg or {})
   if opt.num_batches ~= '' then
      opt.num_batches = loadstring('return ' .. opt.num_batches)()
   else
      opt.num_batches = nil
   end

   if opt.samples_dir == 'SAVEFILE' then
      _, opt.samples_dir = path.splitpath(opt.savefile)
   end
   return opt
end

function M.default()
   return cmd:default()
end

function M.anyStartsWith(opt, start)
   for flag, value in pairs(opt) do
      if type(value) == 'boolean' and stringx.startswith(flag, start) and value then
         return true
      end
   end
   return false
end

return M
