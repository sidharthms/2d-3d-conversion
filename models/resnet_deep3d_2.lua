local nn = require 'nn'
local debugger = require 'fb.debugger'
local list = require 'pl.List'
require 'cunn'
require 'cudnn'
require 'rnn'
require 'dpnn'
require 'nngraph'
require 'PaddedUnfold'
require 'Break'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = cudnn.SpatialBatchNormalization

nngraph.setDebug(true)

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

local function createFeaturesModel(opt)
   cutorch.setDevice(1)
   local resnet_model = torch.load(opt.resnet_model)

   local input = nn.Identity()()
   local initLayersNode = seqFromModules(list.slice(resnet_model.modules, 1, 4))(input)

   local layer1node = resnet_model.modules[5](initLayersNode)
   local layer2node = resnet_model.modules[6](layer1node)
   local layer3node = resnet_model.modules[7](layer2node)
   local layer4node = resnet_model.modules[8](layer3node)

   local featuresNode = nn.Identity()({layer1node, layer2node, layer3node, layer4node})

   local featuresModule = nn.GPU(nn.gModule({input}, {featuresNode}), 1)

   nngraph.annotateNodes()
   return featuresModule:cuda()
end

local function createModel(opt)
   cutorch.setKernelPeerToPeerAccess(true)
   local featureModule = createFeaturesModel(opt)

   local kwFactor = 1
   local pad = 0
   local outChannels = opt.disp_hidden_size

   cutorch.setDevice(2)
   local upsampleNet = nn.ParallelTable()
   for i=1, 4 do
      local convUpsampleNet = nn.Sequential()
      convUpsampleNet:add(bottleneck(64 * 2^(i-1)))
      convUpsampleNet:add(Convolution(64 * 2^(i-1) * 4,outChannels,1,1))
      convUpsampleNet:add(cudnn.SpatialFullConvolution(outChannels, outChannels,
            kwFactor * 2^(i-1), kwFactor * 2^(i-1), 2^(i-1), 2^(i-1), pad, pad))

      upsampleNet:add(convUpsampleNet)
   end

   local dispMapNet = nn.Sequential()
   dispMapNet:add(upsampleNet)
   dispMapNet:add(nn.CAddTable())
   if opt.batchnorm_after_upsample then
      dispMapNet:add(SBatchNorm(outChannels))
   end
   if opt.relu_after_upsample then
      dispMapNet:add(ReLU(true))
   end
   dispMapNet:add(cudnn.SpatialFullConvolution(outChannels, opt.disparity_range,
         opt.disp_upsample_range * 4, opt.disp_upsample_range * 4, 4, 4,
         (opt.disp_upsample_range-1) * 2, (opt.disp_upsample_range-1) * 2))
   dispMapNet:add(nn.Transpose({2, 4}))
   dispMapNet:add(nn.Bottle(cudnn.SoftMax()))
   dispMapNet:add(nn.Transpose({2, 4}))
   dispMapNet:add(nn.Replicate(3, 2))

   local gpuSeq = nn.ParallelTable():add(nn.Identity()):add(dispMapNet)

   if opt.kaimint_init then
      kaimingInit(gpuSeq)
   end
   local gpuNet = nn.GPU(gpuSeq, 2):cuda()

   -- CPU

   local imgUnfoldNet = nn.Sequential()
   imgUnfoldNet:add(PaddedUnfold(4, (opt.disparity_range-1)/2))

   local cpuSeq = nn.Sequential()
   cpuSeq:add(nn.ParallelTable():add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor')):add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor')))
   cpuSeq:add(nn.ParallelTable():add(imgUnfoldNet):add(nn.Identity()))
   cpuSeq:add(nn.CMulTable())
   cpuSeq:add(nn.Sum(3))

   local model = nn.Sequential():add(gpuSeq):add(cpuSeq)

   nngraph.annotateNodes()

   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet then
      print('Optimizing mem')
      local optnet = require 'optnet'
      local optbatchsize = 4
      local sampleInput = torch.zeros(optbatchsize,3,opt.ysize,opt.xsize):cuda()
      local sampleFeatures = {
         torch.zeros(optbatchsize,256,opt.ysize/4,opt.xsize/4):cuda(),
         torch.zeros(optbatchsize,512,opt.ysize/8,opt.xsize/8):cuda(),
         torch.zeros(optbatchsize,1024,opt.ysize/16,opt.xsize/16):cuda(),
         torch.zeros(optbatchsize,2048,opt.ysize/32,opt.xsize/32):cuda(),
      }
      optnet.optimizeMemory(featureModule, sampleInput, {inplace = false, mode = 'training'})
      optnet.optimizeMemory(gpuSeq, {sampleInput, sampleFeatures}, {inplace = false, mode = 'training'})
   end

   return model, featureModule
end

return createModel
