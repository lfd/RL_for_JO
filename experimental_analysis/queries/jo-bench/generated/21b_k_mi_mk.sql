SELECT * FROM movie_keyword AS mk, movie_info AS mi, keyword AS k WHERE k.keyword = 'sequel' AND mi.info IN ('Germany', 'German') AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id;