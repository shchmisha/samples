from ENV import Environment


env = Environment("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")

if __name__ == '__main__':
    # 1) create user
    # env.create_user('user', '123456')
    # 2) login
    user_id = env.login('user', '123456')
    # print(user_id)

    # # 3) upload doc
    env.upload_doc(user_id, 'rbm')

    # # 4) list docs
    # docs = env.list_user_docs(user_id)
    # print(docs)

    # 5) make request to reformat the doc:
    # print(env.make_request(user_id, 'svd', 'explain the svd'))

    # 6) make request to answer a question:
    print(env.make_request('cbd963c6-4cea-4d51-b008-63b5e2f5a4e2', 'rbm', 'what is a restricted boltzmann machine'))