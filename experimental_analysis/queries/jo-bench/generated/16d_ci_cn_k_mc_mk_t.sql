SELECT * FROM company_name AS cn, movie_companies AS mc, keyword AS k, movie_keyword AS mk, title AS t, cast_info AS ci WHERE cn.country_code = '[us]' AND k.keyword = 'character-name-in-title' AND t.episode_nr >= 5 AND t.episode_nr < 100 AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id;