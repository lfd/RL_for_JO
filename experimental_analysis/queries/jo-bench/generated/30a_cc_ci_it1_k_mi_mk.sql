SELECT * FROM info_type AS it1, complete_cast AS cc, cast_info AS ci, keyword AS k, movie_keyword AS mk, movie_info AS mi WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it1.info = 'genres' AND k.keyword IN ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND mi.info IN ('Horror', 'Thriller') AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;