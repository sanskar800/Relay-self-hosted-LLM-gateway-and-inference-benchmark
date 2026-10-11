-- Token buckets for Relay, executed atomically inside Redis.
--
-- Two buckets per tenant: requests and LLM tokens. A bucket holds up to `capacity`
-- tokens and refills continuously at `rate` per second. State per bucket is a hash
-- {tokens, ts}. Time comes from Redis (TIME), so every gateway instance uses the same
-- clock. A missing key means a full bucket; idle keys expire.
--
-- KEYS[1] requests bucket, KEYS[2] tokens bucket
-- ARGV: op ("acquire" | "settle"),
--       req_capacity, req_rate, tok_capacity, tok_rate, amount, ttl_seconds
--   acquire: take 1 request AND `amount` tokens, or take nothing.
--   settle:  add `amount` (may be negative) to the tokens bucket, capped at capacity.
-- Returns {allowed (1/0), req_tokens, tok_tokens, retry_after_s} as strings
-- (Lua numbers would be truncated to integers on the way back to the client).

local time = redis.call('TIME')
local now = tonumber(time[1]) + tonumber(time[2]) / 1000000

local op = ARGV[1]
local req_cap, req_rate = tonumber(ARGV[2]), tonumber(ARGV[3])
local tok_cap, tok_rate = tonumber(ARGV[4]), tonumber(ARGV[5])
local amount = tonumber(ARGV[6])
local ttl = tonumber(ARGV[7])

local function current(key, capacity, rate)
  local state = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens, ts = tonumber(state[1]), tonumber(state[2])
  if tokens == nil or ts == nil then
    return capacity
  end
  return math.min(capacity, tokens + math.max(0, now - ts) * rate)
end

local function save(key, tokens)
  redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', tostring(now))
  redis.call('EXPIRE', key, ttl)
end

local req = current(KEYS[1], req_cap, req_rate)
local tok = current(KEYS[2], tok_cap, tok_rate)

if op == 'settle' then
  tok = math.min(tok_cap, tok + amount)
  save(KEYS[2], tok)
  return {'1', tostring(req), tostring(tok), '0'}
end

local allowed = req >= 1 and tok >= amount
local retry_after = 0
if allowed then
  req = req - 1
  tok = tok - amount
else
  -- Seconds until BOTH buckets could cover this request.
  if req < 1 then retry_after = (1 - req) / req_rate end
  if tok < amount then retry_after = math.max(retry_after, (amount - tok) / tok_rate) end
end
save(KEYS[1], req)
save(KEYS[2], tok)
return {allowed and '1' or '0', tostring(req), tostring(tok), tostring(retry_after)}
