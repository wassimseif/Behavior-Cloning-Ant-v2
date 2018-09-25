import tensorflow as tf
import numpy as np
import pickle
# tf.enable_eager_execution()
import gym
import tensorflow.keras.models
tf.reset_default_graph()

def load_dataset():
    with open('expert_data/Ant-v2.pkl', 'rb') as f:
        data = pickle.loads(f.read())
    return  data

data = load_dataset()

observations  = data['observations']
observations = np.array(observations)
observations = observations.reshape((19057, 111))
# observations  = tf.convert_to_tensor(observations)
batch_size = observations.shape[1]

labels = data['actions']
labels = np.array(labels)
labels = labels.reshape(19057,8)
# labels  = tf.convert_to_tensor(labels)
# Observations

with tf.variable_scope(name_or_scope='Input'):
    X = tf.placeholder(dtype=tf.float32,shape = [batch_size,111])

# Actions
with tf.variable_scope(name_or_scope='Actions_Placeholder'):

    Y = tf.placeholder(dtype=tf.float32,shape = [batch_size,8])

# # Hidden neurons
with tf.variable_scope(name_or_scope='Hidden_1'):
    hidden  = tf.layers.dense(X,units=128,activation=tf.nn.relu)

with tf.variable_scope(name_or_scope='Hidden_2'):
    hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu)


with tf.variable_scope(name_or_scope='Logits'):
    # Make output layers
    logits = tf.layers.dense(hidden, 8)


with tf.variable_scope(name_or_scope='Actions'):
    # Take the action with the highest activation

    action = tf.nn.softmax(logits=logits,axis=1)


with tf.variable_scope(name_or_scope='Loss'):
    # onehot_labels = tf.one_hot(indices=tf.cast(Y, tf.int32), depth=0)  # 4 actions
    # tf.losses.softmax_cross_entropy
    # loss = tf.losses.softmax_cross_entropy(onehot_labels=Y,logits=logits)

    loss =  tf.reduce_mean(tf.losses.mean_squared_error(Y,logits))
    tf.summary.scalar('loss', loss)


with tf.variable_scope(name_or_scope='training'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_op = optimizer.minimize(loss=loss)



env = gym.make('Ant-v2')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("output", sess.graph)
while True:
    done = False

    obs = env.reset()
    steps = 0
    while not done:
        env.render()

        batch_index = np.random.choice(len(observations), size=(batch_size))  # Batch size

        state_batch, action_batch = observations[batch_index], labels[batch_index]

        _ ,cur_loss,curr_summ = sess.run([train_op, loss,merged], feed_dict={
            X: state_batch,
            Y: action_batch

            })
        steps += 1
        writer.add_summary(curr_summ)
        if steps == 600 :
            print("Loss: {}".format(cur_loss))
            steps = 0



        action1 = sess.run(action,feed_dict={
            X: state_batch,
            Y: action_batch
        })


        if steps == 0 :
            print(action1)


        obs, reward, done, info = env.step(action1)


