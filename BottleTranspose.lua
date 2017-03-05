local BottleTranspose, parent = torch.class("BottleTranspose", "nn.Container")
local unpack = unpack or table.unpack

function BottleTranspose:__init(module, nInputDim, nOutputDim, ...)
   parent.__init(self)
   self.nInputDim = nInputDim or 2
   self.nOutputDim = nOutputDim or self.nInputDim
   self.dimDelta = self.nInputDim - self.nOutputDim
   -- Used to reshape the gradients
   self.inShape = torch.Tensor(self.nInputDim)
   self.outShape = torch.Tensor(self.nOutputDim)
   -- add module to modules
   self.modules[1] = module
   self.permutations = {...}
end

function BottleTranspose:updateOutput(input)
   for _,perm in ipairs(self.permutations) do
      input = input:transpose(perm[1],perm[2]):contiguous()
   end

   -- first batchDims dimensions will be fused
   local batchDims = input:dim() - self.nInputDim + 1
   -- see if bottle is required
   if batchDims > 1 then
      -- bottle the first dims
      local inSize = torch.LongTensor(input:size())
      local squeezeSize = inSize[{{1, batchDims - 1}}]:prod()
      self.inShape:copy(inSize[{{batchDims, input:dim()}}])
      self.inShape[{{1}}]:mul(squeezeSize)
      -- Forward with the module's dimension
      local newInput = input:view(unpack(self.inShape:totable()))
      local output = self.modules[1]:updateOutput(newInput)
      assert(output:dim() == self.nOutputDim,
	     "Wrong number of output dims on module. Expected: " ..
		self.nOutputDim .. ' but got ' ..
		tostring(output and output:dim()))
      self.outShape:copy(torch.LongTensor(output:size()))
      if math.abs(self.dimDelta) > 0 then
         inSize:resize(inSize:size(1) - self.dimDelta)
      end
      inSize[{{batchDims, inSize:size(1)}}]:copy(self.outShape)
      inSize[{{batchDims}}]:div(squeezeSize)
      -- unbottle
      self.output:set(output:view(unpack(torch.totable(inSize))))
   else
      self.output:set(self.modules[1]:updateOutput(input))
   end

   for i = #self.permutations,1,-1 do
      local perm = self.permutations[i]
      self.output = self.output:transpose(perm[1],perm[2]):contiguous()
   end

   return self.output
end

function BottleTranspose:updateGradInput(input, gradOutput)
   for _,perm in ipairs(self.permutations) do
      input = input:transpose(perm[1],perm[2]):contiguous()
      gradOutput = gradOutput:transpose(perm[1],perm[2]):contiguous()
   end

   if input:dim() > self.nInputDim then
      local input_ = input:view(unpack(self.inShape:totable()))
      local gradOutput_ = gradOutput:view(unpack(self.outShape:totable()))
      self.modules[1]:updateGradInput(input_, gradOutput_)
      self.gradInput:set(self.modules[1].gradInput:viewAs(input))
   else
      self.gradInput:set(self.modules[1]:updateGradInput(input))
   end

   for i = #self.permutations,1,-1 do
      local perm = self.permutations[i]
      self.gradInput = self.gradInput:transpose(perm[1],perm[2]):contiguous()
   end

   return self.gradInput
end

function BottleTranspose:accGradParameters(input, gradOutput, scale)
   for _,perm in ipairs(self.permutations) do
      input = input:transpose(perm[1],perm[2]):contiguous()
      gradOutput = gradOutput:transpose(perm[1],perm[2]):contiguous()
   end

   if input:dim() > self.nInputDim then
      input = input:view(unpack(self.inShape:totable()))
      gradOutput = gradOutput:view(unpack(self.outShape:totable()))
   end
   self.modules[1]:accGradParameters(input, gradOutput, scale)
end
