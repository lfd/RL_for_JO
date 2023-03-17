SELECT * FROM keyword AS k, movie_keyword AS mk, cast_info AS ci, name AS n WHERE k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'fight') AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;