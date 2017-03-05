local PaddedUnfold, parent = torch.class('PaddedUnfold', 'nn.Module')

function PaddedUnfold:__init(shiftDim, shiftRange)
	parent.__init(self)
	self.shiftDim = shiftDim
	self.shiftRange = shiftRange

	self.padded = torch.Tensor()
	self.gradPadded = torch.Tensor()
end

function PaddedUnfold:updateOutput(input)
	local width = input:size(self.shiftDim)

	local leftPad = input:narrow(self.shiftDim, 1, self.shiftRange)
	local rightPad = input:narrow(self.shiftDim, width-self.shiftRange+1, self.shiftRange)
	torch.cat(self.padded, {leftPad, input, rightPad}, self.shiftDim)

	self.output = self.padded:unfold(self.shiftDim, width, 1)
	assert(self.output:size(self.shiftDim) == self.shiftRange*2+1)
   self.output = self.output:transpose(3, 4)

	-- self.output:resize(self.shiftRange*2+1, table.unpack(input:size()))
	-- for i=1, self.shiftRange*2+1 do
	-- 	local outslice = self.output:select(self.shiftDim, i):transpose(3, 4)
	-- 	assert(outslice:eq(self.padded:narrow(self.shiftDim, i, width)))
	-- end
	-- print(#self.output)
	-- debugger.enter()

	-- self.output = self.output:permute(2, 3, 4, 5, 1)
	return self.output
end

function PaddedUnfold:updateGradInput(input, gradOutput)
	gradOutput = gradOutput:transpose(3, 4)

	-- local m1 = cutorch.getMemoryUsage(2)
	local width = input:size(self.shiftDim)

	self.gradPadded:resizeAs(self.padded):zero()
	for i=1, self.shiftRange*2+1 do
		local gradOutSlice = gradOutput:select(self.shiftDim, i):transpose(3, 4)
		self.gradPadded:narrow(self.shiftDim, i, width):add(gradOutSlice)
	end

	local gradLeftPad = self.gradPadded:narrow(self.shiftDim, 1, self.shiftRange)
	local gradRightPad = self.gradPadded:narrow(self.shiftDim,
	self.gradPadded:size(self.shiftDim)-self.shiftRange+1, self.shiftRange)
	-- print('check gradPadded bounds')
	-- debugger.enter()

	local gradCenter = self.gradPadded:narrow(self.shiftDim, self.shiftRange+1, width)
	self.gradInput:resize(input:size()):copy(gradCenter)
	self.gradInput:narrow(self.shiftDim, 1, self.shiftRange):add(gradLeftPad)
	self.gradInput:narrow(self.shiftDim, width-self.shiftRange+1, self.shiftRange):add(gradRightPad)
	-- local m2 = cutorch.getMemoryUsage(2)

	return self.gradInput
end

function PaddedUnfold:clearState()
	self.padded:set()
	self.gradPadded:set()
	return parent.clearState(self)
end
