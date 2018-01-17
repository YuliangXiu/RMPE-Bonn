local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints = 15
    self.accIdxs = {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
    self.flipRef = {{1,4}, {2,5}, {3,6},
                    {7,10}, {8,11}, {9,12}}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,14,1}, {2,1,1},
                        {3,2,1}, {4,14,1}, {5,4,1}, {6,5,1},
                        {7,8,2}, {8,9,2},
                        {10,11,2}, {11,12,2},{13,14,3}}

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
        opt.idxRef.train = torch.randperm(annot.part:size(1)-2000)
        opt.idxRef.valid = torch.range(annot.part:size(1)-2000,annot.part:size(1))

        torch.save(opt.save .. '/options.t7', opt)
    end

    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel()}
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:split(pString, pPattern)
   local Table = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pPattern
   local last_end = 1
   local s, e, cap = pString:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
     table.insert(Table,cap)
      end
      last_end = e+1
      s, e, cap = pString:find(fpat, last_end)
   end
   if last_end <= #pString then
      cap = pString:sub(last_end)
      table.insert(Table, cap)
   end
   return Table
end

function Dataset:getPath(idx)
    return self:split(paths.concat(opt.dataDir,'images',ffi.string(self.annot.imgname[idx]:char():data())),"@")[1]
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx),3)
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

