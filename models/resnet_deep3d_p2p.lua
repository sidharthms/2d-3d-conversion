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
   -- s:add(ReLU(true))
   -- s:add(Convolution(n,n*4,1,1,1,1,0,0))
   -- s:add(SBatchNorm(n * 4))

   return s
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

   local featuresModule = nn.GPU(nn.gModule({input}, {featuresNode}), 1, 2)

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
   local bottleneckSeq = nn.ParallelTable()
   for i=1, 4 do
      local seq = nn.Sequential()
      seq:add(bottleneck(64 * 2^(i-1)))
      bottleneckSeq:add(seq)
   end

   local e1 = - nn.Identity()
   local e2 = - nn.Identity()
   local e3 = - nn.Identity()
   local e4 = - nn.Identity()

   local d4_ = e4 - Convolution(64*8, 64*4 * 4, 3,3,1,1,1,1) - nn.PixelShuffle(2) - SBatchNorm(64*4) - ReLU(true)
   local d4 = {d4_, e3} - nn.JoinTable(2)

   local d3_ = d4 - Convolution(64*4 * 2, 64*2 * 4, 3,3,1,1,1,1) - nn.PixelShuffle(2) - SBatchNorm(64*2) - ReLU(true)
   local d3 = {d3_, e2} - nn.JoinTable(2)

   local d2_ = d3 - Convolution(64*2 * 2, 64 * 4, 3,3,1,1,1,1) - nn.PixelShuffle(2) - SBatchNorm(64) - ReLU(true)
   local d2 = {d2_, e1} - nn.JoinTable(2)

   local d1 = d2 - Convolution(64 * 2, outChannels, 3,3,1,1,1,1)

   local upsampleGraph = nn.gModule({e1, e2, e3, e4}, {d1})
   local upsampleSeq = nn.Sequential()
   upsampleSeq:add(bottleneckSeq)
   upsampleSeq:add(upsampleGraph)
   local gpu1seq = nn.ParallelTable():add(nn.Identity()):add(upsampleSeq)

   if opt.kaimint_init then
      kaimingInit(gpu1seq)
   end
   local gpu1net = nn.GPU(gpu1seq, 2):cuda()

   cutorch.setDevice(2)

   local dispMapNet = nn.Sequential()
   if opt.batchnorm_after_upsample then
      dispMapNet:add(SBatchNorm(outChannels))
   end
   if opt.relu_after_upsample then
      dispMapNet:add(ReLU(true))
   end
   -- if opt.magicpony_upsample then
   dispMapNet:add(Convolution(outChannels, opt.disparity_range*4,3,3,1,1,1,1))
   -- if opt.batchnorm_after_upsample then
   --    dispMapNet:add(SBatchNorm(opt.disparity_range*4))
   -- end
   dispMapNet:add(ReLU(true))
   dispMapNet:add(Convolution(opt.disparity_range*4, opt.disparity_range*16,3,3,1,1,1,1))
   dispMapNet:add(nn.PixelShuffle(4))
   -- else
   --    dispMapNet:add(cudnn.SpatialFullConvolution(outChannels, opt.disparity_range,
   --          opt.disp_upsample_range * 4, opt.disp_upsample_range * 4, 4, 4,
   --          (opt.disp_upsample_range-1) * 2, (opt.disp_upsample_range-1) * 2))
   -- end
   -- dispMapNet:add(Convolution(outChannels, opt.disparity_range, 1, 1))
   -- dispMapNet:add(nn.PrintSize('SpatialFullConvolution out'))
   dispMapNet:add(nn.Transpose({2, 4}))
   dispMapNet:add(nn.Bottle(cudnn.SoftMax()))
   dispMapNet:add(nn.Transpose({2, 4}))
   -- dispMapNet:add(Break('SoftMax out'))
   dispMapNet:add(nn.Replicate(3, 2))
   -- dispMapNet:add(nn.PrintSize('dispMapNet out'))
   -- dispMapNet:add(Break('dispMapNet out'))
   -- local dispMap = dispMapNet(upsampled)

   local imgUnfoldNet = nn.Sequential()
   --- verify
   imgUnfoldNet:add(PaddedUnfold(4, (opt.disparity_range-1)/2))
   -- imgUnfoldNet:add(nn.PrintSize('PaddedUnfold out'))
   -- imgUnfoldNet:add(Break('PaddedUnfold out'))
   -- imgUnfoldNet:add(nn.Transpose({3, 4}))
   -- imgUnfoldNet:add(nn.PrintSize('imgUnfold out'))
   -- imgUnfoldNet:add(Break('imgUnforld out'))
   --- verify
   -- local imgUnfold = imgUnfoldNet(img)

   local gpu2seq = nn.Sequential()
   gpu2seq:add(nn.ParallelTable():add(imgUnfoldNet):add(dispMapNet))
   gpu2seq:add(nn.CMulTable())
   gpu2seq:add(nn.Sum(3))

   -- local replicatedDispMap = nn.Replicate(3, 2)(dispMap)
   -- local depthDot = nn.CMulTable()({replicatedDispMap, imgUnfold})
   -- local rightImg = nn.Sum(3)(depthDot)
   -- local
   -- local rightChannels = {}
   -- for c=1, 3 do
   --    local channel = nn.Select(2, c)(imgUnfold)
   --    local depthDot = nn.CMulTable()({dispMap, channel})
   --    local regenNet = nn.Sequential()
   --    regenNet:add(nn.Sum(2))
   --    regenNet:add(nn.Unsqueeze(2))
   --    local regen = regenNet(depthDot)
   --    table.insert(rightChannels, regen)
   -- end

   -- local rightImg = nn.JoinTable(2)(rightChannels)

   if opt.kaimint_init then
      kaimingInit(gpu2seq)
   end
   local gpu2net = nn.GPU(gpu2seq, 2):cuda()
   local model = nn.Sequential():add(gpu1net):add(gpu2net)

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
      optnet.optimizeMemory(model, {sampleInput, sampleFeatures}, {inplace = false, mode = 'training'})
   end

   return model, featureModule
end

return createModel
