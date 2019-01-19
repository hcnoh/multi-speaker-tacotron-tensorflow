import tensorflow as tf

import hyparams as hp


def main():
    model_load_path = hp.model_load_path
    model_save_path = hp.model_load_path

    name_from = "multi_speaker_attention_wrapper"
    name_to = "conditional_attention_wrapper"

    for var_name, size in tf.contrib.framework.list_variables(model_load_path):
        var = tf.contrib.framework.load_variable(hp.model_load_path, var_name)
        #print(var_name)

        new_name = var_name
        if name_from in var_name:
            new_name = new_name.replace(name_from, name_to)
            print("A name \"%s\" is to be changed into \"%s\"..." % (var_name, new_name))

        var = tf.Variable(var, name=new_name)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, model_save_path)


if __name__ == '__main__':
    main()
