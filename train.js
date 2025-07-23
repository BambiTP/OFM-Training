// Wait for the game to initialize
setTimeout(() => {
  if (typeof balls === 'undefined' || balls.length < 2) {
    console.error("Balls not initialized. Ensure training.js is loaded after main script and game has started.");
    return;
  }

  // Global training variables
  window.trainingMode = true;
  window.episode = 0;
  window.step = 0;
  const maxSteps = 1000;  // Max steps per episode
  const maxEpisodes = 100; // Total episodes to train

  // State function: normalized position, velocity, flags, and vector to nearest flag
  function getState(ballIndex) {
    const ball = balls[ballIndex];
    const otherIndex = 1 - ballIndex;
    const other = balls[otherIndex];
    
    const p = ball.body.GetPosition();
    const v = ball.body.GetLinearVelocity();
    const x_norm = p.x / (mapWidthGrid * gridSizeTPU);
    const y_norm = p.y / (mapHeightGrid * gridSizeTPU);
    const vx_norm = v.x / playerProperties.maxSpeed;
    const vy_norm = v.y / playerProperties.maxSpeed;
    
    const other_p = other.body.GetPosition();
    const other_x_norm = other_p.x / (mapWidthGrid * gridSizeTPU);
    const other_y_norm = other_p.y / (mapHeightGrid * gridSizeTPU);
    
    const own_flag = ball.flag ? 1 : 0;
    const other_flag = other.flag ? 1 : 0;
    
    let minDist = Infinity;
    let nearestFlag = null;
    for (let b = world.GetBodyList(); b; b = b.GetNext()) {
      const fixture = b.GetFixtureList();
      if (fixture) {
        const userData = fixture.GetUserData();
        if (userData && userData.tile === 'YellowFlag' && userData.state === 1) {
          const flag_p = b.GetPosition();
          const dx = flag_p.x - p.x;
          const dy = flag_p.y - p.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < minDist) {
            minDist = dist;
            nearestFlag = flag_p;
          }
        }
      }
    }
    let dx_norm = 0, dy_norm = 0;
    if (nearestFlag) {
      dx_norm = (nearestFlag.x - p.x) / (mapWidthGrid * gridSizeTPU);
      dy_norm = (nearestFlag.y - p.y) / (mapHeightGrid * gridSizeTPU);
    }
    
    return [x_norm, y_norm, vx_norm, vy_norm, other_x_norm, other_y_norm, own_flag, other_flag, dx_norm, dy_norm];
  }

  // Reward function: +1 if holding flag, -1 otherwise
  function getReward(ballIndex) {
    return balls[ballIndex].flag ? 1 : -1;
  }

  // Action function: map action index to input combinations (9 actions)
  function setInputFromAction(ball, action) {
    ball.input.up = false;
    ball.input.down = false;
    ball.input.left = false;
    ball.input.right = false;
    
    switch (action) {
      case 1: ball.input.up = true; break;
      case 2: ball.input.down = true; break;
      case 3: ball.input.left = true; break;
      case 4: ball.input.right = true; break;
      case 5: ball.input.up = true; ball.input.left = true; break;
      case 6: ball.input.up = true; ball.input.right = true; break;
      case 7: ball.input.down = true; ball.input.left = true; break;
      case 8: ball.input.down = true; ball.input.right = true; break;
      // case 0: no movement
    }
  }

  // Reset function: reset balls and flags
  function reset() {
    for (let id = 0; id < 2; id++) {
      const ball = balls[id];
      ball.flag = false;
      ball.canControl = true;
      ball.sprite.visible = true;
      ball.body.SetActive(true);
      let tx = (id === 0 ? 23 : 1) * gridSizeTPU + gridSizeTPU / 2;  // Blue at (23,1), Red at (1,17)
      let ty = (id === 0 ? 1 : 17) * gridSizeTPU + gridSizeTPU / 2;
      ball.body.SetPosition(new b2Vec2(tx, ty));
      ball.body.SetLinearVelocity(new b2Vec2(0, 0));
      ball.body.SetAngularVelocity(0);
      ball.body.SetAngle(0);
    }
    
    for (let b = world.GetBodyList(); b; b = b.GetNext()) {
      const fixture = b.GetFixtureList();
      if (fixture) {
        const userData = fixture.GetUserData();
        if (userData && userData.tile === 'YellowFlag') {
          userData.state = 1;
          const key = `${userData.gridX},${userData.gridY}`;
          const sprite = midground.children.find(s => s._gridKey === key);
          if (sprite) sprite.alpha = 1;
        }
      }
    }
  }

  // DQN Agent class
  class DQNAgent {
    constructor() {
      this.model = this.createModel();
      this.targetModel = this.createModel();
      this.targetModel.setWeights(this.model.getWeights());
      this.replayBuffer = [];
      this.epsilon = 1.0;
      this.epsilonDecay = 0.995;
      this.minEpsilon = 0.01;
      this.gamma = 0.99;
      this.batchSize = 32;
      this.updateTargetFrequency = 100;
      this.stepCount = 0;
    }

    createModel() {
      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [10] }));
      model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
      model.add(tf.layers.dense({ units: 9 }));
      model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
      return model;
    }

    act(state) {
      if (Math.random() < this.epsilon) {
        return Math.floor(Math.random() * 9);
      }
      const stateTensor = tf.tensor2d([state]);
      const qValues = this.model.predict(stateTensor);
      const action = qValues.argMax(1).dataSync()[0];
      stateTensor.dispose();
      qValues.dispose();
      return action;
    }

    learn(state, action, reward, nextState, done) {
      this.replayBuffer.push({ state, action, reward, nextState, done });
      if (this.replayBuffer.length > 10000) this.replayBuffer.shift();
      if (this.replayBuffer.length < this.batchSize) return;

      const batch = this.sampleBatch();
      const states = batch.map(e => e.state);
      const nextStates = batch.map(e => e.nextState);
      const statesTensor = tf.tensor2d(states);
      const nextStatesTensor = tf.tensor2d(nextStates);
      const qValues = this.model.predict(statesTensor);
      const nextQValues = this.targetModel.predict(nextStatesTensor);
      const qValuesData = qValues.dataSync();
      const nextQValuesData = nextQValues.dataSync();

      for (let i = 0; i < batch.length; i++) {
        const { action, reward, done } = batch[i];
        const qIndex = i * 9 + action;
        const nextQSlice = nextQValuesData.slice(i * 9, (i + 1) * 9);
        const target = done ? reward : reward + this.gamma * Math.max(...nextQSlice);
        qValuesData[qIndex] = target;
      }

      const updatedQValues = tf.tensor2d(qValuesData, [this.batchSize, 9]);
      this.model.trainOnBatch(statesTensor, updatedQValues);

      this.stepCount++;
      if (this.stepCount % this.updateTargetFrequency === 0) {
        this.targetModel.setWeights(this.model.getWeights());
      }
      this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);

      statesTensor.dispose();
      nextStatesTensor.dispose();
      qValues.dispose();
      nextQValues.dispose();
      updatedQValues.dispose();
    }

    sampleBatch() {
      const indices = Array.from({ length: this.batchSize }, () => Math.floor(Math.random() * this.replayBuffer.length));
      return indices.map(i => this.replayBuffer[i]);
    }
  }

  // Initialize agents
  const agent0 = new DQNAgent();
  const agent1 = new DQNAgent();

  // Override update for training
  let prevStates, actions;
  const originalUpdate = update;
  update = function() {
    if (trainingMode) {
      prevStates = [getState(0), getState(1)];
      actions = [agent0.act(prevStates[0]), agent1.act(prevStates[1])];
      setInputFromAction(balls[0], actions[0]);
      setInputFromAction(balls[1], actions[1]);
    }
    originalUpdate();
    if (trainingMode) {
      const nextStates = [getState(0), getState(1)];
      const rewards = [getReward(0), getReward(1)];
      const done = (step >= maxSteps);
      agent0.learn(prevStates[0], actions[0], rewards[0], nextStates[0], done);
      agent1.learn(prevStates[1], actions[1], rewards[1], nextStates[1], done);
      step++;
      if (done) {
        episode++;
        step = 0;
        reset();
        if (episode >= maxEpisodes) {
          trainingMode = false;
          console.log(`Training completed after ${maxEpisodes} episodes`);
        } else {
          console.log(`Episode ${episode} started`);
        }
      }
    }
  };

  // Start training with initial reset
  console.log("Starting training...");
  reset();
}, 1000); // Delay to ensure game initialization
