local nn = require 'nn'
local debugger = require 'fb.debugger'
local list = require 'pl.List'
require 'cunn'
require 'cudnn'
require 'rnn'
require 'dpnn'
require 'PaddedUnfold'
require 'Break'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = cudnn.SpatialBatchNormalization

-- The bottleneck residual layer for 50, 101, and 152 layer networks
local function bottleneck(n)
   local nInputPlane = n * 4

   local s = nn.Sequential()
   s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
   s:add(SBatchNorm(n))
   s:add(ReLU(true))
   s:add(Convolution(n,n,3,3,1,1,1,1))
   s:add(SBatchNorm(n))
   s:add(ReLU(true))
   s:add(Convolution(n,n*4,1,1,1,1,0,0))
   s:add(SBatchNorm(n * 4))

   return nn.Sequential()
      :add(nn.ConcatTable()
         :add(s)
         :add(nn.Identity()))
      :add(nn.CAddTable(true))
      :add(ReLU(true))
end

local function seqFromModules(modules)
   local seq = nn.Sequential()
   for m=1, #modules do
      seq:add(modules[m])
   end
   return seq
end

local function createFeaturesModel(opt, baseModelCheckpoint)
   cutorch.setDevice(1)
   local seq = nn.Sequential()
   seq:add(baseModelCheckpoint.featureModel)
   seq:add(baseModelCheckpoint.model.modules[1].modules[1].modules[2])

   local dispMapNet = baseModelCheckpoint.model.modules[2].modules[1].modules[1].modules[2]
   local lastIndex
   for i, m in ipairs(dispMapNet.modules) do
      if torch.typename(m) == 'nn.Transpose' then
         lastIndex = i-1
         break
      end
   end
   seq:add(seqFromModules(list.slice(dispMapNet.modules, 1, lastIndex)))

   local featuresModule = nn.GPU(seq, 1)

   nngraph.annotateNodes()
   return featuresModule:cuda()
end

local function createModel(opt)
   -- cutorch.setKernelPeerToPeerAccess(true)
   local baseModelCheckpoint = torch.load(opt.deep3d_base)
   local featureModule = createFeaturesModel(opt, baseModelCheckpoint)

   local kwFactor = 1
   local pad = 0
   local outChannels = opt.disp_hidden_size

   cutorch.setDevice(2)

   local kW = opt.magicpony_highres_upsample_kernel
   local dispMapNet = nn.Sequential()
   dispMapNet:add(Convolution(opt.disparity_range, opt.disparity_range * 4, kW,kW,1,1,(kW-1)/2,(kW-1)/2))
   if opt.highres_batchnorm then
      dispMapNet:add(SBatchNorm(n))
   end
   dispMapNet:add(ReLU(true))
   dispMapNet:add(Convolution(opt.disparity_range * 4, opt.disparity_range * 16, kW,kW,1,1,(kW-1)/2,(kW-1)/2))
   dispMapNet:add(nn.PixelShuffle(4))
   dispMapNet:add(nn.Transpose({2, 4}))
   dispMapNet:add(nn.Bottle(cudnn.SoftMax()))
   dispMapNet:add(nn.Transpose({2, 4}))
   dispMapNet:add(nn.Replicate(3, 2))

   local imgUnfoldNet = nn.Sequential()
   imgUnfoldNet:add(PaddedUnfold(4, (opt.disparity_range-1)/2))

   local seq = nn.Sequential()
   seq:add(nn.ParallelTable():add(imgUnfoldNet):add(dispMapNet))
   seq:add(nn.CMulTable())
   seq:add(nn.Sum(3))

   if opt.kaimint_init then
      kaimingInit(seq)
   end
   local gpu2net = nn.GPU(seq, 2):cuda()
   local model = seq

   nngraph.annotateNodes()

   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet then
      print('Optimizing mem')
      local optnet = require 'optnet'
      local optbatchsize = 4
      local sampleInput = torch.zeros(optbatchsize, 3, opt.ysize, opt.xsize):cuda()
      local sampleFeatures = torch.zeros(optbatchsize, opt.disparity_range,
            baseModelCheckpoint.opt.ysize, baseModelCheckpoint.opt.xsize):cuda()
      optnet.optimizeMemory(featureModule, sampleInput, {inplace = false, mode = 'training'})
      optnet.optimizeMemory(model, {sampleInput, sampleFeatures}, {inplace = false, mode = 'training'})
   end

   return model, featureModule
end

return createModel
