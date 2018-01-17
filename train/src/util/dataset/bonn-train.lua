local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints = 14
    self.accIdxs = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
    self.flipRef = {{1,6}, {2,5}, {3,4},
                    {7,12}, {8,11}, {9,10}}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,14,1}, {2,1,1},
                        {3,2,1}, {6,14,1}, {5,6,1}, {4,5,1},
                        {7,8,2}, {8,9,2},
                        {12,11,2}, {11,10,2},{13,14,3},{14,15,3}}

    local annot = {}
    local tags = {'imgname','part','bndbox'}
    local a = hdf5.open('../data/'..opt.dataset..'/annot.h5','r')
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    a:close()
    annot.part:add(1)

    -- Index reference
    if not opt.idxRef then
        opt.idxRef = {}
        -- Set up training/validation split
        opt.idxRef.train = torch.randperm(annot.part:size(1)-558)
        opt.idxRef.valid = torch.range(annot.part:size(1)-558,annot.part:size(1))

        torch.save(opt.save .. '/options.t7', opt)
    end

    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel()}
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
    return paths.concat(opt.dataDir,'images',ffi.string(self.annot.imgname[idx]:char():data()))
end

function Dataset:loadImage(idx)
    -- print(idx, self:getPath(idx):sub(1,84))
    return image.load(self:getPath(idx):sub(1,84),3)
end

function Dataset:getPartInfo(idx)
    local pts = self.annot.part[idx]:clone()
    local bndbox = self.annot.bndbox[idx]:clone()

    return pts, bndbox
end

function Dataset:getNormInfo(idx)
    local mu = {-0.0142, 0.0043, 0.0154, -0.0013}
    local std = {0.1158, 0.068, 0.1337, 0.0711}

    return mu, std
end


return M.Dataset

