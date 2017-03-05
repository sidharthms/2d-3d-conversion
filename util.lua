--
-- Created by IntelliJ IDEA.
-- User: sidharth
-- Date: 3/20/16
-- Time: 10:59 PM
-- To change this template use File | Settings | File Templates.
--
local debugger = require 'fb.debugger'

function standard_eh(err)
    print(err)
    debugger.enter()
end

function ConvInit(model, name)
   for k,v in pairs(model:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if cudnn.version >= 4000 then
         v.bias = nil
         v.gradBias = nil
      else
         v.bias:zero()
      end
   end
end
function BNInit(model, name)
   for k,v in pairs(model:findModules(name)) do
      v.weight:fill(1)
      v.bias:zero()
   end
end

function kaimingInit(model)
   ConvInit(model, 'cudnn.SpatialConvolution')
   ConvInit(model, 'nn.SpatialConvolution')
   BNInit(model, 'fbnn.SpatialBatchNormalization')
   BNInit(model, 'cudnn.SpatialBatchNormalization')
   BNInit(model, 'nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
end
