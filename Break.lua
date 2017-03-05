local Break, parent = torch.class('Break', 'nn.Module')

function Break:__init(prefix, forward, backward)
   parent.__init(self)
   self.forwardBreak = forward or true
   self.backwardBreak = backward or true
   self.prefix = prefix or "Break"
end

function Break:updateOutput(input)
   self.output = input
   if self.forwardBreak then
      print(self.prefix..":break")
      debugger.enter()
   end
   return self.output
end


function Break:updateGradInput(input, gradOutput)
   if self.backwardBreak then
      print(self.prefix..":gradOutput")
      debugger.enter()
   end
   self.gradInput = gradOutput
   return self.gradInput
end
