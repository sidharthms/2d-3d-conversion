--
--  This code is based on https://github.com/facebook/fb.resnet.torch
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
local tablex = require('pl.tablex')
local listx = require('pl.List')
local dir = require 'pl.dir'
local path = require 'pl.path'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

local function clip_gradients(grads, max_grad_norm, max_grad)
    local grad_norm = torch.norm(grads)
    local scale_num = max_grad_norm
    local scale_den = math.max(max_grad_norm, grad_norm)
    grads = grads * (scale_num / scale_den)
    if grads:max() > 10 then
        print('factor = ', (scale_num / scale_den))
    end
    grads = torch.clamp(grads, -max_grad, max_grad)
    return grads
end

function Trainer:__init(model, featureModel, criterion, opt)
    self.model = model
    self.featureModel = featureModel
    self.criterion = criterion
    if opt.optim == 'sgd' then
        self.optimState = {
            learningRate = opt.lr,
            learningRateDecay = 0.0,
            momentum = opt.momentum,
            nesterov = true,
            dampening = 0.0,
            weightDecay = opt.weight_decay
        }
    elseif opt.optim == 'adam' then
        self.optimState = {
            learningRate = opt.lr,
            weightDecay = opt.weight_decay
        }
    else
        error('Unknown optim type')
    end
    self.opt = opt
    self.params, self.rawGradParams = eh.on_error(model.getParameters, standard_eh, model)
    print('Size gradParams:', self.rawGradParams:size())

    if self.opt.batch_size_factor ~= 1 then
        self.gradParams = self.rawGradParams:clone():zero()
    else
        self.gradParams = self.rawGradParams
    end

    -- Clear old output images
    local sample_dir = path.join(self.opt.data_dir, self.opt.samples_dir)
    if self.opt.samples_dir ~= '' and path.exists(sample_dir) then
        dir.rmtree(sample_dir)
    end
end

function Trainer:getFeval()
    -- debugger.enter()
    return function()
        local f = self.criterion.output

        -- clip gradients
        local clipped_gradients = self.gradParams

        if self.opt.clip_grads then
            clipped_gradients = clip_gradients(self.gradParams, self.opt.grad_max_norm, self.opt.grad_max)
        end

        local stats_str = ('f: %.3f, max: %.3f, min: %.3f, mean: %.3f; gmax: %.3f, gmin: %.3f'
                .. ', gmean: %.3f'):format(f, self.params:max(), self.params:min(), self.params:mean(),
                clipped_gradients:max(), clipped_gradients:min(), clipped_gradients:mean())

        if not self.opt.no_print_stats then
            print(stats_str)
        end

        return f, clipped_gradients
    end
end

function Trainer:should_skip(batch)
    -- if batch < 42 then
    --     return true
    -- end
    return false
end

function Trainer:run(train, epoch, dataloader, split_idx, threshold, mistakes_filename, batch_callback, batch_limit,
        predictions_filename)
    assert(train ~= nil, 'train can\'t be nil')
    assert(split_idx ~= nil, 'Split index can\'t be nil')
    if train then
        assert(split_idx == 1, 'Cannot train over a data set other than training set')

        -- Trains the model for a single epoch
        self.optimState.learningRate = self:learningRate(epoch)
        print('Learing rate for epoch ' .. epoch .. ': ' .. self.optimState.learningRate)
    end

    local set_type
    if split_idx == 1 then
        set_type = 'Train'
    elseif split_idx == 2 then
        set_type = 'Validation'
    else
        set_type = 'Test'
    end

    local run_type
    if train then
        run_type = 'TRAINING'
    else
        run_type = 'EVALUATING'
    end

    local timer = torch.Timer()
    local dataTimer = torch.Timer()

    local totalDataTime = 0
    local lossSum = 0.0
    local tpSum = 0.0
    local fpSum = 0.0
    local fnSum = 0.0
    local N = 0
    local batches = dataloader:batches(split_idx)
    local batch_limit = batch_limit and batch_limit > 0 and batch_limit or batches
    local all_outputs = {}

    print('=> ' .. run_type .. ' Epoch # ' .. epoch)
    -- set the batch norm to training or evaluation mode
    if train then
        self.model:training()
    else
        self.model:evaluate()
    end

    for batch, x, y, d in dataloader:run(split_idx, self.opt.batch_randperm) do
        if not self:should_skip(batch) then
            totalDataTime = totalDataTime + dataTimer:time().real

            -- x = x:sub(1, 48)
            -- y = y:sub(1, 48)

            -- self.model:clearState()
            self:copyInputs(x, y, d)

            -- if batch == 1 then
            --     local features = eh.on_error(self.featureModel.forward, standard_eh, self.featureModel, self.input)
            --     local raw_out = eh.on_error(self.model.forward, standard_eh, self.model, {self.input, features})
            -- end

            local raw_out
            local features = self.featureModel:forward(self.input)
            raw_out = self.model:forward({self.input, features})
            -- print('Verify out')
            if self.opt.samples_dir ~= '' and batch % self.opt.sample_save_rate == 0 then
                dir.makepath(path.join(self.opt.data_dir, self.opt.samples_dir))
                local suffix = '_' .. set_type .. '_' .. tostring(epoch) .. '_' .. tostring(batch)
                save_img(path.join(self.opt.data_dir, self.opt.samples_dir, 'input' .. suffix .. '.ppm'), self.input[1])
                save_img(path.join(self.opt.data_dir, self.opt.samples_dir, 'prediction' .. suffix .. '.ppm'), raw_out[1])
                save_img(path.join(self.opt.data_dir, self.opt.samples_dir, 'target' .. suffix .. '.ppm'), self.target[1])
            end
            local loss = 100 * self.criterion:forward(self.model.output, self.target)
            lossSum = lossSum + loss
            N = N + 1

            if self.opt.unmonitored_mode then
                assert(self.params:ne(self.params):sum() == 0, 'NaN found in params')
            elseif self.params:ne(self.params):sum() > 0 then
                print('nan detected')
                debugger.enter()
            end

            -- local err = self:computeScore(raw_out, y, threshold)

            local doGradUpdate = false
            if train then
                self.model:zeroGradParameters()
                self.criterion:backward(self.model.output, self.target)
                self.model:backward(self.input, self.criterion.gradInput)
                -- debugger.enter()

                if self.opt.batch_size_factor ~= 1 then
                    batch_phase = (N-1) % self.opt.batch_size_factor + 1
                    if batch_phase == 1 then
                        self.gradParams:zero()
                    end
                    -- debugger.enter()
                    self.gradParams:add(self.rawGradParams)

                    if batch_phase == self.opt.batch_size_factor or batch == batch_limit then
                        -- debugger.enter()
                        self.gradParams:div(batch_phase)
                        doGradUpdate = true
                    end
                else
                    doGradUpdate = true
                end

                if doGradUpdate then
                    if self.opt.unmonitored_mode then
                        assert(self.gradParams:ne(self.gradParams):sum() == 0, 'NaN found in gradparams')
                    elseif self.gradParams:ne(self.gradParams):sum() > 0 then
                        print('nan detected')
                        debugger.enter()
                    end
                    if self.opt.optim == 'sgd' then
                        optim.sgd(self:getFeval(), self.params, self.optimState)
                    elseif self.opt.optim == 'adam' then
                        optim.adam(self:getFeval(), self.params, self.optimState)
                    else
                        error('Unknown optim method')
                    end
                end
            end
            -- debugger.enter()

            print((' | ' .. set_type .. ' : [%d][%d/%d]     loss %7.3f (%7.3f)'):format(
                epoch, batch, batches, loss, lossSum / N))

            if doGradUpdate and self.opt.batch_size_factor ~= 1 then
                print('Performed grad update...')
            end

            -- if batch % 100 == 0 then
            --     -- Run model on validation set
            --     self:validation(epoch, dataloader)
            --     print('Continuing TRAINING')
            -- end

            -- check that the storage didn't get changed do to an unfortunate getParameters call
            assert(self.params:storage() == self.model:parameters()[1]:storage())

            dataTimer:reset()

            if batch_callback then
                batch_callback(x, y, d, output)
            end
        elseif batch > batch_limit then
            break
        else
            print((' | ' .. set_type .. ' : [%d][%d/%d]     === SKIPPED ==='):format(epoch, batch, batches))
        end
    end
    -- self.model:training()

    local lossAvg = lossSum / N
    print(('Finished Epoch: [%d]  Time %.3f  Data %.3f  Err: %7.3f\n'):format(
            epoch, timer:time().real, totalDataTime, lossAvg))

    return lossAvg
end

function Trainer:computeScore(output, target, threshold)
    if self.opt.noscore then
        return math.huge, math.huge, math.huge, math.huge
    end
    -- Coputes the err
    local batchSize = output:size(1)

    threshold = threshold or 0.5
    local _ , predictions = output:float():sort(2, true) -- descending
    local predictions2 = output:narrow(2, 1, 1):float():lt(threshold) + 1
    predictions = predictions:narrow(2, 1, 1)

--    print('predictions : ', predictions)
--    print('targets : ', target)

    local target_long = target:long()
    -- Find which predictions match the target
    -- debugger.enter()

    -- Sanity check targets
    assert(target:le(self.opt.max_class_idx):all())
    assert(target:ge(1):all())

    local correct = predictions:eq(target:long())

    local ones = torch.ByteTensor(correct:size()):fill(1)
    local positives = torch.eq(torch.Tensor(target_long:size()):long():fill(2), target_long)
    local tp = correct:dot(positives)
    local fn = (ones - correct):dot(positives)
    local fp = (ones - correct):dot(ones - positives)

    local err = 1.0 - correct:narrow(2, 1, 1):sum() / batchSize

    return err * 100, tp, fn, fp
end

function Trainer:copyInputs(x, y, d)
    -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
    -- if using DataParallelTable. The target is always copied to a CUDA tensor
    self.input = self.input or torch.CudaTensor()
    self.target = self.target or torch.CudaTensor()
    self.input:resize(x:size()):copy(x)
    self.target:resize(y:size()):copy(y)
end

function Trainer:learningRate(epoch)
    -- Training schedule
    local decay = 0
    if epoch ~= 1 then
        decay = math.floor((epoch - 1) / self.opt.lr_decay_period)
    end
    return self.opt.lr * math.pow(self.opt.lr_decay_factor, decay)
end

return M.Trainer
