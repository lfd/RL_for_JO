SELECT * FROM keyword AS k, movie_keyword AS mk, cast_info AS ci WHERE k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'fight') AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;